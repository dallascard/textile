import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import KFold

from ..util import file_handling as fh
from ..models import lr, mlp, evaluation, calibration, ensemble
from ..preprocessing import features
from ..util import dirs
from ..util.misc import printv


def main():
    usage = "%prog project_dir subset model_name config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|BLR|MLP]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--ensemble', action="store_true", dest="ensemble", default=False,
                      help='Make an ensemble from cross-validation, instead of training one model: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--weights', dest='weights_file', default=None,
                      help='Weights file: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    model_name = args[2]
    config_file = args[3]

    model_type = options.model
    do_ensemble = options.ensemble
    dh = int(options.dh)
    label = options.label
    weights_file = options.weights_file
    penalty = options.penalty
    objective = options.objective
    intercept = not options.no_intercept
    n_dev_folds = int(options.n_dev_folds)
    if options.seed is not None:
        np.random.seed(int(options.seed))

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file, penalty=penalty, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed)


def train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file=None, items_to_use=None,
                penalty='l2', alpha_min=0.01, alpha_max=1000, n_alphas=8, intercept=True,
                objective='f1', n_dev_folds=5, save_model=True, do_ensemble=False, dh=0, seed=None, verbose=True):

    label_dir = dirs.dir_labels(project_dir, subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    weights = None
    #if weights_file is not None:
    #    weights_df = fh.read_csv_to_df(weights_file)
    #    assert np.all(weights_df.index == labels_df.index)
    #    weights = weights_df['weight'].values

    return train_model_with_labels(project_dir, model_type, model_name, subset, labels_df, feature_defs, weights,
                                   items_to_use, penalty, alpha_min, alpha_max, n_alphas, intercept, objective,
                                   n_dev_folds, save_model, do_ensemble, dh, seed, verbose)


def train_model_with_labels(project_dir, model_type, model_name, subset, labels_df, feature_defs, weights_df=None,
                            items_to_use=None, penalty='l2', alpha_min=0.01, alpha_max=1000, n_alphas=8, intercept=True,
                            objective='f1', n_dev_folds=5, save_model=True, do_ensemble=False, dh=0, seed=None,
                            verbose=True):

    features_dir = dirs.dir_features(project_dir, subset)
    n_items, n_classes = labels_df.shape

    indices_to_use = None
    if items_to_use is not None:
        item_index = dict(zip(labels_df.index, range(n_items)))
        indices_to_use = [item_index[i] for i in items_to_use]
        labels_df = labels_df.loc[items_to_use]
        n_items, n_classes = labels_df.shape
    else:
        items_to_use = list(labels_df.index)

    if weights_df is not None:
        weights = np.array(weights_df.loc[items_to_use].values).reshape(n_items,)
    else:
        weights = np.ones(n_items)
    print("weights shape", weights.shape)

    printv("loading features", verbose)
    feature_list = []
    feature_signatures = []
    for feature_def in feature_defs:
        printv(feature_def, verbose)
        name = feature_def.name
        feature = features.load_from_file(input_dir=features_dir, basename=name)
        # take a subset of the rows, if requested
        printv("Initial shape = (%d, %d)" % feature.get_shape(), verbose)
        if indices_to_use is not None:
            printv("Taking subset of items", verbose)
            feature = features.create_from_feature(feature, indices_to_use)
            printv("New shape = (%d, %d)" % feature.get_shape(), verbose)
        feature.threshold(feature_def.min_df)
        if feature_def.transform == 'doc2vec':
            word_vectors_prefix = os.path.join(features_dir, name + '_vecs')
        else:
            word_vectors_prefix = None
        feature.transform(feature_def.transform, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
        printv("Final shape = (%d, %d)" % feature.get_shape(), verbose)
        feature_list.append(feature)
        if save_model:
            feature_signature = features.get_feature_signature(feature_def, feature)
            # save the location of the word vectors from training... (need a better solution for this eventually)
            feature_signature['word_vectors_prefix'] = word_vectors_prefix
            feature_signatures.append(feature_signature)

    output_dir = os.path.join(dirs.dir_models(project_dir), model_name)
    if save_model:
        fh.makedirs(output_dir)
        fh.write_to_json(feature_signatures, os.path.join(output_dir, 'features.json'), sort_keys=False)

    features_concat = features.concatenate(feature_list)
    col_names = features_concat.terms

    if features_concat.sparse:
        X = features_concat.get_counts().tocsr()
    else:
        X = features_concat.get_counts()
    Y = labels_df.as_matrix()

    n_items, n_features = X.shape
    _, n_classes = Y.shape

    #weights = pd.DataFrame(1.0/labels_df.sum(axis=1), index=labels_df.index, columns=['inv_n_labels'])
    # divide weights by the number of annotations that we have for each item
    #weights = weights * 1.0/Y.sum(axis=1)

    print("Train feature matrix shape: (%d, %d)" % X.shape)

    try:
        assert np.array(features_concat.items == labels_df.index).all()
    except AssertionError:
        print("mismatch in items between labels and features")
        print(features_concat.items[:5])
        print(labels_df.index[:5])
        sys.exit()

    kfold = KFold(n_splits=n_dev_folds, shuffle=True)
    if n_alphas > 1:
        alpha_factor = np.power(alpha_max / alpha_min, 1.0/(n_alphas-1))
        alphas = np.array(alpha_min * np.power(alpha_factor, np.arange(n_alphas)))
    else:
        alphas = [alpha_min]

    mean_train_f1s = np.zeros(n_alphas)
    mean_dev_f1s = np.zeros(n_alphas)
    mean_dev_acc = np.zeros(n_alphas)
    mean_dev_cal = np.zeros(n_alphas)
    mean_model_size = np.zeros(n_alphas)
    acc_cfms = []
    pvc_cfms = []

    print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('iter', 'alpha', 'size', 'f1_trn', 'f1_dev', 'acc_dev', 'dev_cal'))

    model = None
    model_ensemble = None
    if do_ensemble:
        model_ensemble = ensemble.Ensemble(output_dir, model_name)

    if model_type == 'LR':
        for alpha_i, alpha in enumerate(alphas):
            alpha_acc_cfms = []
            alpha_pvc_cfms = []
            fold = 1
            for train_indices, dev_indices in kfold.split(X):
                model = lr.LR(alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name='temp')

                X_train = X[train_indices, :]
                Y_train = Y[train_indices, :]
                X_dev = X[dev_indices, :]
                Y_dev = Y[dev_indices, :]
                w_train = weights[train_indices]
                w_dev = weights[dev_indices]
                X_train, Y_train, w_train = expand_features_and_labels(X_train, Y_train, w_train)
                X_dev, Y_dev, w_dev = expand_features_and_labels(X_dev, Y_dev, w_dev)

                model.fit(X_train, Y_train, train_weights=w_train, X_dev=X_dev, Y_dev=Y_dev, dev_weights=w_dev, col_names=col_names)

                train_predictions = model.predict(X_train)
                dev_predictions = model.predict(X_dev)

                y_train_vector = np.argmax(Y_train, axis=1)
                y_dev_vector = np.argmax(Y_dev, axis=1)

                # internally compute the correction matrices
                alpha_acc_cfms.append(calibration.compute_acc(y_dev_vector, dev_predictions, n_classes, weights=w_dev))
                alpha_pvc_cfms.append(calibration.compute_pvc(y_dev_vector, dev_predictions, n_classes, weights=w_dev))

                train_f1 = evaluation.f1_score(y_train_vector, train_predictions, n_classes, weights=w_train)
                dev_f1 = evaluation.f1_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
                train_acc = evaluation.acc_score(y_train_vector, train_predictions, n_classes, weights=w_train)
                dev_acc = evaluation.acc_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
                dev_cal_rmse = evaluation.evaluate_proportions_mse(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
                #evaluation.evaluate_calibration_mse_bins(y[dev_indices], dev_predictions, 1)

                mean_train_f1s[alpha_i] += train_f1 / float(n_dev_folds)
                mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
                mean_dev_acc[alpha_i] += dev_acc / float(n_dev_folds)
                mean_dev_cal[alpha_i] += dev_cal_rmse / float(n_dev_folds)

                mean_model_size[alpha_i] += model.get_model_size() / float(n_dev_folds)

            print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i], mean_dev_acc[alpha_i], mean_dev_cal[alpha_i]))

            acc_cfms.append(np.mean(alpha_acc_cfms, axis=0))
            pvc_cfms.append(np.mean(alpha_pvc_cfms, axis=0))

        if objective == 'f1':
            best_alpha_index = mean_dev_f1s.argmax()
            print("Using best f1: %d" % best_alpha_index)
        elif objective == 'calibration':
            best_alpha_index = mean_dev_cal.argmin()
            print("Using best calibration: %d" % best_alpha_index)
        else:
            sys.exit("Objective not recognized")
        best_alpha = alphas[best_alpha_index]
        best_dev_f1 = mean_dev_f1s[best_alpha_index]
        best_dev_cal = mean_dev_cal[best_alpha_index]
        print("Best: alpha = %.3f, dev f1 = %.3f, dev cal = %.3f" % (best_alpha, best_dev_f1, best_dev_cal))

        best_acc_cfm = acc_cfms[best_alpha_index]
        best_pvc_cfm = pvc_cfms[best_alpha_index]

        if do_ensemble:
            printv("Retraining with best alpha for ensemble", verbose)
            fold = 1
            for train_indices, dev_indices in kfold.split(X):
                name = model_name + '_' + str(fold)
                model = lr.LR(best_alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name=name)
                X_train = X[train_indices, :]
                Y_train = Y[train_indices, :]
                X_dev = X[dev_indices, :]
                Y_dev = Y[dev_indices, :]
                w_train = weights[train_indices]
                w_dev = weights[dev_indices]
                X_train, Y_train, w_train = expand_features_and_labels(X_train, Y_train, w_train)
                X_dev, Y_dev, w_dev = expand_features_and_labels(X_dev, Y_dev, w_dev)
                model.fit(X_train, Y_train, train_weights=w_train, X_dev=X_dev, Y_dev=Y_dev, dev_weights=w_dev, col_names=col_names, seed=seed)
                if save_model:
                    model.save()
                model_ensemble.add_model(model, name)
                fold += 1
            model = model_ensemble
            model.save()

        else:
            printv("Training full model", verbose)
            model = lr.LR(best_alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name=model_name)

            X, Y, w = expand_features_and_labels(X, Y, weights)
            model.fit(X, Y, train_weights=w, col_names=col_names)
            model.save()

    elif model_type == 'MLP':
        if dh > 0:
            dimensions = [n_features, dh, n_classes]
        else:
            dimensions = [n_features, n_classes]
        if not save_model:
            output_dir = None

        fold = 1
        for train_indices, dev_indices in kfold.split(X):
            if do_ensemble:
                name = model_name + '_' + str(fold)
            else:
                name = model_name
            model = mlp.MLP(dimensions=dimensions, loss_function='log', nonlinearity='tanh', penalty=penalty, reg_strength=0, output_dir=output_dir, name=name)

            X_train = X[train_indices, :]
            Y_train = Y[train_indices, :]
            X_dev = X[dev_indices, :]
            Y_dev = Y[dev_indices, :]
            w_train = weights[train_indices]
            w_dev = weights[dev_indices]
            X_train, Y_train, w_train = expand_features_and_labels(X_train, Y_train, w_train)
            X_dev, Y_dev, w_dev = expand_features_and_labels(X_dev, Y_dev, w_dev)

            model.fit(X_train, Y_train, X_dev, Y_dev, train_weights=w_train, dev_weights=w_dev)
            if save_model:
                model.save()

            if do_ensemble:
                model_ensemble.add_model(model, name)
                fold += 1
            else:
                break

        if do_ensemble:
            model = model_ensemble
            model.save()

    else:
        sys.exit("Model type %s not recognized" % model_type)

    """
    elif model_type == 'BLR':
        printv("Fitting single model with ARD", verbose)
        model = blr.BLR(alpha=None, fit_intercept=intercept, n_classes=n_classes)
        model.fit(np.array(X.todense()), y, col_names, sample_weights=weights, batch=True, multilevel=True, ard=True)
        best_dev_f1 = 0
        best_dev_cal = 0
        best_acc_cfm = None
        best_pvc_cfm = None
    """

    return model


def expand_features_and_labels(X, Y, weights=None, predictions=None):
    """
    Expand the feature matrix and label matrix by converting items with multiple labels to multiple rows with 1 each
    :param X: feature matrix (n_items, n_features)
    :param Y: label matrix (n_items, n_classes)
    :param weights: (n_items, )
    :return: 
    """
    X_list = []
    Y_list = []
    weights_list = []
    pred_list = []
    n_items, n_classes = Y.shape
    if weights is None:
        weights = np.ones(n_items)
    for i in range(n_items):
        labels = Y[i, :]
        total = labels.sum()
        for index, count in enumerate(labels):
            if count > 0:
                for c in range(count):
                    X_list.append(X[i, :])
                    label_vector = np.zeros(n_classes, dtype=int)
                    label_vector[index] = 1
                    Y_list.append(label_vector)
                    weights_list.append(weights[i] * 1.0/total)
                    if predictions is not None:
                        pred_list.append(predictions[i])

    if sparse.issparse(X):
        X_return = sparse.vstack(X_list)
    else:
        X_return = np.vstack(X_list)
    y_return = np.array(Y_list)
    weights_return = np.array(weights_list)
    if predictions is None:
        return X_return, y_return, weights_return
    else:
        pred_return = np.array(pred_list)
        return X_return, y_return, weights_return, pred_return


if __name__ == '__main__':
    main()
