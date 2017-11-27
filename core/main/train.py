import os
import sys
from optparse import OptionParser
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import distance
from sklearn.model_selection import KFold

from ..util import file_handling as fh
from ..models import linear, evaluation, ensemble
from ..models import decision_list, dan
from ..models import get_top_features
from ..preprocessing import features
from ..util import dirs
from ..util.misc import printv


def main():
    usage = "%prog project_dir subset model_name config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|DAN]: default=%default')
    parser.add_option('--loss', dest='loss', default='log',
                      help='Loss function [log|brier]: default=%default')
    parser.add_option('--dh', dest='dh', default=0,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--no_ensemble', action="store_true", dest="no_ensemble", default=False,
                      help='Do not use an ensemble: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--weights', dest='weights_file', default=None,
                      help='Weights file: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
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
    loss = options.loss
    do_ensemble = not options.no_ensemble
    dh = int(options.dh)
    label = options.label
    weights_file = options.weights_file
    penalty = options.penalty
    objective = options.objective
    intercept = not options.no_intercept
    n_dev_folds = int(options.n_dev_folds)
    seed = options.seed
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    train_model(project_dir, model_type, loss, model_name, subset, label, feature_defs, weights_file, penalty=penalty, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed)


def train_model(project_dir, model_type, loss, model_name, subset, label, feature_defs, items_to_use=None,
                penalty='l2', alpha_min=0.01, alpha_max=1000, n_alphas=8, intercept=True,
                objective='f1', n_dev_folds=5, save_model=True, do_ensemble=True, dh=0,
                seed=None, verbose=True):

    label_dir = dirs.dir_labels(project_dir, subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    weights = None

    return train_model_with_labels(project_dir, model_type, loss, model_name, subset, labels_df, feature_defs, weights,
                                   items_to_use, penalty, alpha_min, alpha_max, n_alphas, intercept, objective,
                                   n_dev_folds, save_model, do_ensemble, dh, seed, verbose)


def train_model_with_labels(project_dir, model_type, loss, model_name, subset, labels_df, feature_defs, weights_df=None,
                            items_to_use=None, penalty='l1', alpha_min=0.01, alpha_max=1000, n_alphas=8, intercept=True,
                            objective='f1', n_dev_folds=5, save_model=True, do_ensemble=True, dh=0, seed=None,
                            pos_label=1, vocab=None, group_identical=False, nonlinearity='tanh',
                            init_lr=1e-3, min_epochs=2, max_epochs=100, patience=8,
                            list_size=10, do_cfm=False, do_platt=False, dl_feature_list=None,
                            lower=None, interactive=False, stoplist=None,
                            update_emb=False, dropout=0.0, verbose=True):

    features_dir = dirs.dir_features(project_dir, subset)
    n_items, n_classes = labels_df.shape

    if items_to_use is not None:
        item_index = dict(zip(labels_df.index, range(n_items)))
        #indices_to_use = [item_index[i] for i in items_to_use]
        labels_df = labels_df.loc[items_to_use]
        n_items, n_classes = labels_df.shape
    else:
        items_to_use = list(labels_df.index)

    if weights_df is not None:
        weights = np.array(weights_df.loc[items_to_use].values).reshape(n_items,)
    else:
        weights = np.ones(n_items)

    vocab_index = None
    if vocab is not None:
        vocab_index = dict(zip(vocab, range(len(vocab))))

    init_embeddings = None
    printv("loading features", verbose)
    feature_list = []
    feature_signatures = []
    for feature_def in feature_defs:
        printv(feature_def, verbose)
        name = feature_def.name
        feature = features.load_from_file(input_dir=features_dir, basename=name)
        # take a subset of the rows, if requested
        printv("Initial shape = (%d, %d)" % feature.get_shape(), verbose)
        feature_items = feature.get_items()
        feature_item_index = dict(zip(feature_items, range(len(feature_items))))
        #printv("len(feature_items)" + str(len(feature_items)), verbose)
        #printv("len(items_to_use)" + str(len(items_to_use)), verbose)
        #printv(str(list(np.sum(i for i in items_to_use if i in feature_item_index))), verbose)
        indices_to_use = [feature_item_index[i] for i in items_to_use]
        if indices_to_use is not None:
            printv("Taking subset of items", verbose)
            feature = features.create_from_feature(feature, indices_to_use)
            printv("New shape = (%d, %d)" % feature.get_shape(), verbose)
        feature.threshold(feature_def.min_df)
        if vocab is not None:
            feature_vocab = [term for term in feature.get_terms() if term in vocab_index]
            feature.set_terms(feature_vocab)
        if stoplist is not None:
            print("Applying stoplist")
            print(len(feature.get_terms()))
            feature_vocab = [term for term in feature.get_terms() if term not in stoplist]
            feature.set_terms(feature_vocab)
            print(len(feature.get_terms()))
        if feature_def.transform == 'doc2vec':
            word_vectors_prefix = os.path.join(features_dir, name + '_vecs')
        else:
            word_vectors_prefix = None
        feature.transform(feature_def.transform, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
        if model_type == 'DAN':
            # TODO: make this less hacky (i.e. deal with non unigram features or something...)
            print(name)
            print(feature.get_shape(), "before embeddings")
            print("Loading word vectors")
            word_vectors_prefix = os.path.join(features_dir, 'unigrams' + '_vecs')
            init_embeddings = np.array(fh.load_dense(word_vectors_prefix + '.npz'), dtype=np.float32)
            word_vector_terms = fh.read_json(word_vectors_prefix + '.json')
            feature.set_terms(word_vector_terms)
            print(feature.get_shape(), "after embeddings")

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

    col_names = features_concat.get_col_names()
    if len(col_names) < 200:
        print(col_names)

    if features_concat.sparse:
        X = features_concat.get_counts().tocsr()
    else:
        X = features_concat.get_counts()
    Y_orig = labels_df.as_matrix()
    n_items, n_features = X.shape

    if group_identical:
        # get the positive proportion for each feature vector
        X_counts = defaultdict(int)
        X_n_pos = defaultdict(int)

        for i in range(n_items):
            if sparse.issparse(X):
                vector = np.array(X[i, :].todense()).ravel()
            else:
                vector = np.array(X[i, :]).ravel()
            key = ''.join([str(int(s)) for s in vector])
            X_counts[key] += int(np.sum(Y_orig[i, :]))
            X_n_pos[key] += int(Y_orig[i, 1])

        keys = list(X_counts.keys())
        keys.sort()
        key_probs = {}

        for key in keys:
            key_probs[key] = X_n_pos[key] / float(X_counts[key])

        """
        for i in range(n_features):
            vector = np.zeros(n_features)
            vector[i] = 1
            key = ''.join([str(int(s)) for s in vector])
            if key in key_probs:
                print(key, key_probs[key])
        """

        fh.write_to_json({'X_counts': X_counts, 'X_n_pos': X_n_pos}, os.path.join(output_dir, 'training_patterns.json'))

    Y = Y_orig

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
    mean_dev_cal_mae = np.zeros(n_alphas)  # track the calibration across the range of probabilities (using bins)
    mean_dev_cal_est = np.zeros(n_alphas)  # track the calibration overall
    mean_model_size = np.zeros(n_alphas)

    print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ('iter', 'alpha', 'size', 'f1_trn', 'f1_dev', 'acc_dev', 'dev_cal_mae', 'dev_cal_est'))

    model = None
    model_ensemble = None
    if do_ensemble:
        model_ensemble = ensemble.Ensemble(output_dir, model_name)

    # store everything, as we'll want it after doing CV
    alpha_models = {}
    best_models = None

    train_splits = []
    dev_splits = []

    for train_indices, dev_indices in kfold.split(X):
        train_splits.append(list(train_indices))
        dev_splits.append(list(dev_indices))

    if model_type == 'LR':
        for alpha_i, alpha in enumerate(alphas):
            alpha_models[alpha] = []

            #for train_indices, dev_indices in kfold.split(X):
            for fold in range(n_dev_folds):
                train_indices = train_splits[fold]
                dev_indices = dev_splits[fold]
                name = model_name + '_' + str(fold)
                model = linear.LinearClassifier(alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name=name, pos_label=pos_label, do_cfm=do_cfm, do_platt=do_platt, lower=lower)

                X_train = X[train_indices, :]
                Y_train = Y[train_indices, :]
                X_dev = X[dev_indices, :]
                Y_dev = Y[dev_indices, :]
                w_train = weights[train_indices]
                w_dev = weights[dev_indices]
                #X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
                #X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

                model.fit(X_train, Y_train, train_weights=w_train, col_names=col_names)

                X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
                X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

                train_predictions = model.predict(X_train)
                dev_predictions = model.predict(X_dev)
                dev_pred_probs = model.predict_probs(X_dev)

                alpha_models[alpha].append(model)
                #print("Adding model to list for %.4f; new length = %d" % alpha, len(alpha_models[alpha]))

                y_train_vector = np.argmax(Y_train, axis=1)
                y_dev_vector = np.argmax(Y_dev, axis=1)

                train_f1 = evaluation.f1_score(y_train_vector, train_predictions, n_classes, pos_label=pos_label, weights=w_train)
                dev_f1 = evaluation.f1_score(y_dev_vector, dev_predictions, n_classes, pos_label=pos_label, weights=w_dev)
                dev_acc = evaluation.acc_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
                dev_proportions = evaluation.compute_proportions(Y_dev, w_dev)
                pred_proportions = evaluation.compute_proportions(dev_pred_probs, w_dev)
                dev_cal_mae = evaluation.eval_proportions_mae(dev_proportions, pred_proportions)
                dev_cal_est = evaluation.evaluate_calibration_rmse(y_dev_vector, dev_pred_probs)

                mean_train_f1s[alpha_i] += train_f1 / float(n_dev_folds)
                mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
                mean_dev_acc[alpha_i] += dev_acc / float(n_dev_folds)
                mean_dev_cal_mae[alpha_i] += dev_cal_mae / float(n_dev_folds)
                mean_dev_cal_est[alpha_i] += dev_cal_est / float(n_dev_folds)
                mean_model_size[alpha_i] += model.get_model_size() / float(n_dev_folds)

            print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i], mean_dev_acc[alpha_i], mean_dev_cal_mae[alpha_i], mean_dev_cal_est[alpha_i]))

        if objective == 'f1':
            best_alpha_index = mean_dev_f1s.argmax()
            print("Using best f1: %d" % best_alpha_index)
        elif objective == 'calibration':
            best_alpha_index = mean_dev_cal_est.argmin()
            print("Using best calibration: %d" % best_alpha_index)
        else:
            sys.exit("Objective not recognized")

        best_alpha = alphas[best_alpha_index]
        best_dev_f1 = mean_dev_f1s[best_alpha_index]
        best_dev_acc = mean_dev_acc[best_alpha_index]
        best_dev_cal_mae = mean_dev_cal_mae[best_alpha_index]
        best_dev_cal_est = mean_dev_cal_est[best_alpha_index]
        print("Best: alpha = %.3f, dev f1 = %.3f, dev cal mae = %.3f, dev calibration estimate= %0.3f" % (best_alpha, best_dev_f1, best_dev_cal_mae, best_dev_cal_est))

        best_models = alpha_models[best_alpha]

        if do_cfm or do_platt:
            for fold, model in enumerate(best_models):
                dev_indices = dev_splits[fold]
                X_dev = X[dev_indices, :]
                Y_dev = Y[dev_indices, :]
                w_dev = weights[dev_indices]
                X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)
                if do_cfm:
                    printv("Fitting cfms", verbose)
                    model.fit_cfms(X_dev, Y_dev, w_dev)
                if do_platt:
                    printv("Fitting Platt scaling", verbose)
                    model.fit_platt(X_dev, Y_dev, w_dev)

        if save_model:
            print("Saving models")
            for model in best_models:
                model.save()

        if do_ensemble:
            printv("Retraining with best alpha for ensemble", verbose)
            fold = 1
            for model_i, model in enumerate(best_models):
                name = model_name + '_' + str(fold)
                model_ensemble.add_model(model, name)
                fold += 1
            full_model = model_ensemble
            if save_model:
                full_model.save()

        else:
            printv("Training full model", verbose)
            full_model = linear.LinearClassifier(best_alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name=model_name, pos_label=pos_label, lower=lower)
            X, Y, w = prepare_data(X, Y, weights, loss=loss)
            full_model.fit(X, Y, train_weights=w, col_names=col_names)
            if save_model:
                full_model.save()

    elif model_type == 'DAN':
        if dh > 0:
            dimensions = [n_features, dh, dh, dh, 1]
        else:
            dimensions = [n_features, 1]
        if not save_model:
            output_dir = None

        best_models = []
        best_dev_f1 = 0.0
        best_dev_acc = 0.0
        best_dev_cal_mae = 0.0
        best_dev_cal_est = 0.0

        for alpha_i, alpha in enumerate(alphas):
            alpha_models[alpha] = []

            fold = 1
            for train_indices, dev_indices in kfold.split(X):
                print("Starting fold %d" % fold)
                name = model_name + '_' + str(fold)
                #model = mlp.MLP(dimensions=dimensions, loss_function=loss, nonlinearity=nonlinearity, penalty=penalty, reg_strength=alpha, output_dir=output_dir, name=name, pos_label=pos_label, objective=objective)

                model = dan.DAN(dimensions, alpha=alpha, output_dir=output_dir, name=name, pos_label=pos_label, objective=objective, init_emb=init_embeddings, update_emb=update_emb)

                X_train = X[train_indices, :]
                Y_train = Y[train_indices, :]
                X_dev = X[dev_indices, :]
                Y_dev = Y[dev_indices, :]
                w_train = weights[train_indices]
                w_dev = weights[dev_indices]
                X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
                X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

                model.fit(X_train, Y_train, X_dev, Y_dev, train_weights=w_train, dev_weights=w_dev, col_names=col_names, seed=seed, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, dropout_prob=dropout)
                alpha_models[alpha].append(model)

                dev_predictions = model.predict(X_dev)
                dev_pred_probs = model.predict_probs(X_dev)
                y_dev_vector = np.argmax(Y_dev, axis=1)

                dev_f1 = evaluation.f1_score(y_dev_vector, dev_predictions, n_classes, pos_label=pos_label, weights=w_dev)
                dev_acc = evaluation.acc_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
                dev_proportions = evaluation.compute_proportions(Y_dev, w_dev)
                pred_proportions = evaluation.compute_proportions(dev_pred_probs, w_dev)
                dev_cal_mae = evaluation.eval_proportions_mae(dev_proportions, pred_proportions)
                dev_cal_est = evaluation.evaluate_calibration_rmse(y_dev_vector, dev_pred_probs)

                mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
                mean_dev_acc[alpha_i] += dev_acc / float(n_dev_folds)
                mean_dev_cal_mae[alpha_i] += dev_cal_mae / float(n_dev_folds)
                mean_dev_cal_est[alpha_i] += dev_cal_est / float(n_dev_folds)
                mean_model_size[alpha_i] += 0
                fold += 1

            print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i], mean_dev_acc[alpha_i], mean_dev_cal_mae[alpha_i], mean_dev_cal_est[alpha_i]))

        if objective == 'f1':
            best_alpha_index = mean_dev_f1s.argmax()
            print("Using best f1: %d" % best_alpha_index)
        elif objective == 'calibration':
            best_alpha_index = mean_dev_cal_est.argmin()
            print("Using best calibration: %d" % best_alpha_index)
        else:
            sys.exit("Objective not recognized")
        best_alpha = alphas[best_alpha_index]
        best_dev_f1 = mean_dev_f1s[best_alpha_index]
        best_dev_acc = mean_dev_acc[best_alpha_index]
        best_dev_cal_mae = mean_dev_cal_mae[best_alpha_index]
        best_dev_cal_est = mean_dev_cal_est[best_alpha_index]
        print("Best: alpha = %.3f, dev f1 = %.3f, dev cal mae = %.3f, dev calibration estimate= %0.3f" % (best_alpha, best_dev_f1, best_dev_cal_mae, best_dev_cal_est))

        best_models = alpha_models[best_alpha]
        print("Number of best models = %d" % len(best_models))

        if save_model:
            print("Saving models")
            for model in best_models:
                model.save()

        if do_ensemble:
            fold = 1
            for model_i, model in enumerate(best_models):
                name = model_name + '_' + str(fold)
                model_ensemble.add_model(model, name)
                fold += 1
            full_model = model_ensemble
            full_model.save()
        else:
            full_model = None

    elif model_type == 'DL':  # decision list

        fold = 1
        alpha = 1
        alpha_i = 0
        alphas[alpha_i] = alpha
        stoplist = []
        alpha_models = {}
        alpha_models[alpha] = []

        for train_indices, dev_indices in kfold.split(X):
            name = model_name + '_' + str(fold)

            # split data
            X_train = X[train_indices, :]
            Y_train = Y[train_indices, :]
            X_dev = X[dev_indices, :]
            Y_dev = Y[dev_indices, :]
            w_train = weights[train_indices]
            w_dev = weights[dev_indices]
            #X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
            #X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

            # train a LR model on half the data
            #lr_model = train_lr_model_with_cv(X_train, Y_train, w_train, col_names, name + '_s1LR', output_dir=output_dir, n_classes=n_classes, objective='f1', loss='log', penalty=penalty, intercept=intercept, n_dev_folds=2, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, pos_label=pos_label, do_ensemble=False, prep_data=False)

            #totals = defaultdict(float)
            #coefs = lr_model.get_coefs(target_class=0)
            #for coef, value in coefs:
            #    totals[coef] += value

            #coef_totals = [(coef, value) for coef, value in totals.items()]
            #coef_totals = sorted(coef_totals, key=lambda x: x[1])
            #coef_totals.reverse()

            #feature_list = [word for word, value in coef_totals]
            #print(feature_list[:25])

            model = decision_list.DL(alpha=1.0, penalty=penalty, output_dir=output_dir, name=name, pos_label=pos_label, max_depth=list_size, fit_intercept=intercept)

            #stoplist = {'of_new_paltz'}
            stoplist = {}
            #feature_list = model.feature_selection(X_train, Y_train, w_train, col_names, max_features=25, stoplist=stoplist)

            model.fit(X_train, Y_train, train_weights=w_train, col_names=col_names, feature_list=dl_feature_list, X_dev=X_dev, Y_dev=Y_dev, dev_weights=w_dev, interactive=interactive, stoplist=stoplist, objective=objective, penalty=penalty, pos_label=pos_label, do_ensemble=do_ensemble)

            X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
            X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

            train_predictions = model.predict(X_train)
            dev_predictions = model.predict(X_dev)
            dev_pred_probs = model.predict_probs(X_dev)

            alpha_models[alpha].append(model)
            #print("Adding model to list for %.4f; new length = %d" % alpha, len(alpha_models[alpha]))

            y_train_vector = np.argmax(Y_train, axis=1)
            y_dev_vector = np.argmax(Y_dev, axis=1)

            # internally compute the correction matrices
            #alpha_acc_cfms.append(calibration.compute_acc(y_dev_vector, dev_predictions, n_classes, weights=w_dev))
            #alpha_pvc_cfms.append(calibration.compute_pvc(y_dev_vector, dev_predictions, n_classes, weights=w_dev))

            train_f1 = evaluation.f1_score(y_train_vector, train_predictions, n_classes, pos_label=pos_label, weights=w_train)
            dev_f1 = evaluation.f1_score(y_dev_vector, dev_predictions, n_classes, pos_label=pos_label, weights=w_dev)
            dev_acc = evaluation.acc_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
            dev_proportions = evaluation.compute_proportions(Y_dev, w_dev)
            pred_proportions = evaluation.compute_proportions(dev_pred_probs, w_dev)
            dev_cal_mae = evaluation.eval_proportions_mae(dev_proportions, pred_proportions)
            dev_cal_est = evaluation.evaluate_calibration_rmse(y_dev_vector, dev_pred_probs)

            print(train_f1, dev_f1, dev_acc, dev_cal_mae, dev_cal_est)

            mean_train_f1s[alpha_i] += train_f1 / float(n_dev_folds)
            mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
            mean_dev_acc[alpha_i] += dev_acc / float(n_dev_folds)
            mean_dev_cal_mae[alpha_i] += dev_cal_mae / float(n_dev_folds)
            mean_dev_cal_est[alpha_i] += dev_cal_est / float(n_dev_folds)
            mean_model_size[alpha_i] += model.get_model_size() / float(n_dev_folds)
            fold += 1

        print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i], mean_dev_acc[alpha_i], mean_dev_cal_mae[alpha_i], mean_dev_cal_est[alpha_i]))

        best_alpha_index = 0
        best_alpha = alphas[best_alpha_index]
        best_dev_f1 = mean_dev_f1s[best_alpha_index]
        best_dev_acc = mean_dev_acc[best_alpha_index]
        best_dev_cal_mae = mean_dev_cal_mae[best_alpha_index]
        best_dev_cal_est = mean_dev_cal_est[best_alpha_index]
        print("Best: alpha = %.3f, dev f1 = %.3f, dev cal mae = %.3f, dev calibration estimate= %0.3f" % (best_alpha, best_dev_f1, best_dev_cal_mae, best_dev_cal_est))

        best_models = alpha_models[best_alpha]
        print("Number of best models = %d" % len(best_models))

        if save_model:
            print("Saving models")
            for model in best_models:
                model.save()

        if save_model:
            print("Saving models")
            for model in best_models:
                model.save()

        printv("Saving ensemble", verbose)
        fold = 1
        for model_i, model in enumerate(best_models):
            name = model_name + '_' + str(fold)
            model_ensemble.add_model(model, name)
            fold += 1
        full_model = model_ensemble
        if save_model:
            full_model.save()

        """ elif model_type == 'MLP':
        if dh > 0:
            dimensions = [n_features, dh, n_classes]
        else:
            dimensions = [n_features, n_classes]
        if not save_model:
            output_dir = None

        best_models = []
        best_dev_f1 = 0.0
        best_dev_acc = 0.0
        best_dev_cal_mae = 0.0
        best_dev_cal_est = 0.0

        if sparse.issparse(X):
            assert X.size < 10000000
            X = np.array(X.todense())

        for alpha_i, alpha in enumerate(alphas):
            alpha_models[alpha] = []

            fold = 1
            for train_indices, dev_indices in kfold.split(X):
                print("Starting fold %d" % fold)
                name = model_name + '_' + str(fold)
                model = mlp.MLP(dimensions=dimensions, loss_function=loss, nonlinearity=nonlinearity, penalty=penalty, reg_strength=alpha, output_dir=output_dir, name=name, pos_label=pos_label, objective=objective)

                X_train = X[train_indices, :]
                Y_train = Y[train_indices, :]
                X_dev = X[dev_indices, :]
                Y_dev = Y[dev_indices, :]
                w_train = weights[train_indices]
                w_dev = weights[dev_indices]
                X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
                X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

                model.fit(X_train, Y_train, X_dev, Y_dev, train_weights=w_train, dev_weights=w_dev, seed=seed, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, tol=tol, early_stopping=early_stopping)
                #best_models.append(model)
                alpha_models[alpha].append(model)

                dev_predictions = model.predict(X_dev)
                dev_pred_probs = model.predict_probs(X_dev)

                y_dev_vector = np.argmax(Y_dev, axis=1)

                dev_f1 = evaluation.f1_score(y_dev_vector, dev_predictions, n_classes, pos_label=pos_label, weights=w_dev)
                dev_acc = evaluation.acc_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
                dev_proportions = evaluation.compute_proportions(Y_dev, w_dev)
                pred_proportions = evaluation.compute_proportions(dev_pred_probs, w_dev)
                dev_cal_mae = evaluation.eval_proportions_mae(dev_proportions, pred_proportions)
                dev_cal_est = evaluation.evaluate_calibration_rmse(y_dev_vector, dev_pred_probs)

                mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
                mean_dev_acc[alpha_i] += dev_acc / float(n_dev_folds)
                mean_dev_cal_mae[alpha_i] += dev_cal_mae / float(n_dev_folds)
                mean_dev_cal_est[alpha_i] += dev_cal_est / float(n_dev_folds)
                mean_model_size[alpha_i] += model.get_model_size() / float(n_dev_folds)
                fold += 1

            print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i], mean_dev_acc[alpha_i], mean_dev_cal_mae[alpha_i], mean_dev_cal_est[alpha_i]))

        if objective == 'f1':
            best_alpha_index = mean_dev_f1s.argmax()
            print("Using best f1: %d" % best_alpha_index)
        elif objective == 'calibration':
            best_alpha_index = mean_dev_cal_est.argmin()
            print("Using best calibration: %d" % best_alpha_index)
        else:
            sys.exit("Objective not recognized")
        best_alpha = alphas[best_alpha_index]
        best_dev_f1 = mean_dev_f1s[best_alpha_index]
        best_dev_acc = mean_dev_acc[best_alpha_index]
        best_dev_cal_mae = mean_dev_cal_mae[best_alpha_index]
        best_dev_cal_est = mean_dev_cal_est[best_alpha_index]
        print("Best: alpha = %.3f, dev f1 = %.3f, dev cal mae = %.3f, dev calibration estimate= %0.3f" % (best_alpha, best_dev_f1, best_dev_cal_mae, best_dev_cal_est))

        best_models = alpha_models[best_alpha]
        print("Number of best models = %d" % len(best_models))

        if save_model:
            print("Saving models")
            for model in best_models:
                model.save()

        if do_ensemble:
            fold = 1
            for model_i, model in enumerate(best_models):
                name = model_name + '_' + str(fold)
                model_ensemble.add_model(model, name)
                fold += 1
            full_model = model_ensemble
            full_model.save()
        else:
            full_model = None
        """
    else:
        sys.exit("Model type %s not recognized" % model_type)

    return full_model, best_dev_f1, best_dev_acc, best_dev_cal_mae, best_dev_cal_est


def train_lr_model_with_cv(X, Y, weights, col_names, basename, output_dir=None, n_classes=2, objective='f1', penalty='l2', intercept=True, n_dev_folds=5, alpha_min=0.01, alpha_max=1000.0, n_alphas=8, pos_label=1, do_ensemble=False, fit_platt=False, fit_cfms=False, save_model=True, lower=None, verbose=True):
    loss = 'log'
    kfold = KFold(n_splits=n_dev_folds, shuffle=True)
    if n_alphas > 1:
        alpha_factor = np.power(alpha_max / alpha_min, 1.0/(n_alphas-1))
        alphas = np.array(alpha_min * np.power(alpha_factor, np.arange(n_alphas)))
    else:
        alphas = [alpha_min]

    mean_train_f1s = np.zeros(n_alphas)
    mean_dev_f1s = np.zeros(n_alphas)
    mean_dev_acc = np.zeros(n_alphas)
    mean_dev_cal_mae = np.zeros(n_alphas)  # track the calibration across the range of probabilities (using bins)
    mean_dev_cal_est = np.zeros(n_alphas)  # track the calibration overall
    mean_model_size = np.zeros(n_alphas)

    alpha_models = {}
    model_ensemble = None
    if do_ensemble:
        model_ensemble = ensemble.Ensemble(output_dir, basename)

    train_splits = []
    dev_splits = []

    for train_indices, dev_indices in kfold.split(X):
        train_splits.append(list(train_indices))
        dev_splits.append(list(dev_indices))

    for alpha_i, alpha in enumerate(alphas):
        alpha_models[alpha] = []

        for fold in range(n_dev_folds):
            train_indices = train_splits[fold]
            dev_indices = dev_splits[fold]

            name = basename + '_temp_' + str(fold)
            model = linear.LinearClassifier(alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name=name, pos_label=pos_label, do_cfm=fit_cfms, do_platt=fit_platt, lower=lower)

            X_train = X[train_indices, :]
            Y_train = Y[train_indices, :]
            w_train = weights[train_indices]
            X_dev = X[dev_indices, :]
            Y_dev = Y[dev_indices, :]
            w_dev = weights[dev_indices]
            #if prep_data:
            #    X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
            #    X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

            model.fit(X_train, Y_train, train_weights=w_train, col_names=col_names)

            X_train, Y_train, w_train = prepare_data(X_train, Y_train, w_train, loss=loss)
            X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)

            train_predictions = model.predict(X_train)
            dev_predictions = model.predict(X_dev)
            dev_pred_probs = model.predict_probs(X_dev)

            alpha_models[alpha].append(model)

            y_train_vector = np.argmax(Y_train, axis=1)
            y_dev_vector = np.argmax(Y_dev, axis=1)

            train_f1 = evaluation.f1_score(y_train_vector, train_predictions, n_classes, pos_label=pos_label, weights=w_train)
            dev_f1 = evaluation.f1_score(y_dev_vector, dev_predictions, n_classes, pos_label=pos_label, weights=w_dev)
            dev_acc = evaluation.acc_score(y_dev_vector, dev_predictions, n_classes, weights=w_dev)
            dev_proportions = evaluation.compute_proportions(Y_dev, w_dev)
            pred_proportions = evaluation.compute_proportions(dev_pred_probs, w_dev)
            dev_cal_mae = evaluation.eval_proportions_mae(dev_proportions, pred_proportions)
            dev_cal_est = evaluation.evaluate_calibration_rmse(y_dev_vector, dev_pred_probs)

            mean_train_f1s[alpha_i] += train_f1 / float(n_dev_folds)
            mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
            mean_dev_acc[alpha_i] += dev_acc / float(n_dev_folds)
            mean_dev_cal_mae[alpha_i] += dev_cal_mae / float(n_dev_folds)
            mean_dev_cal_est[alpha_i] += dev_cal_est / float(n_dev_folds)
            mean_model_size[alpha_i] += model.get_model_size() / float(n_dev_folds)

        if verbose:
            print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f\t%0.3f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i], mean_dev_acc[alpha_i], mean_dev_cal_mae[alpha_i], mean_dev_cal_est[alpha_i]))

    if objective == 'f1':
        best_alpha_index = mean_dev_f1s.argmax()
        if verbose:
            print("Using best f1: %d" % best_alpha_index)
    elif objective == 'calibration':
        best_alpha_index = mean_dev_cal_est.argmin()
        if verbose:
            print("Using best calibration: %d" % best_alpha_index)
    else:
        sys.exit("Objective not recognized")

    best_alpha = alphas[best_alpha_index]
    best_dev_f1 = mean_dev_f1s[best_alpha_index]
    best_dev_cal_mae = mean_dev_cal_mae[best_alpha_index]
    best_dev_cal_est = mean_dev_cal_est[best_alpha_index]
    if verbose:
        print("Best: alpha = %.3f, dev f1 = %.3f, dev cal mae = %.3f, dev calibration estimate= %0.3f" % (best_alpha, best_dev_f1, best_dev_cal_mae, best_dev_cal_est))

    best_models = alpha_models[best_alpha]

    if fit_cfms or fit_platt:
        for fold, model in enumerate(best_models):
            dev_indices = dev_splits[fold]
            X_dev = X[dev_indices, :]
            Y_dev = Y[dev_indices, :]
            w_dev = weights[dev_indices]
            X_dev, Y_dev, w_dev = prepare_data(X_dev, Y_dev, w_dev, loss=loss)
            if fit_cfms:
                model.fit_cfms(X_dev, Y_dev, w_dev)
            if fit_platt:
                model.fit_platt(X_dev, Y_dev, w_dev)

    if do_ensemble:
        fold = 1
        for model_i, model in enumerate(best_models):
            name = basename + '_temp_' + str(fold)
            model_ensemble.add_model(model, name)
            fold += 1
        full_model = model_ensemble

        if save_model:
            for model in best_models:
                model.save()
            full_model.save()

    else:
        if verbose:
            print("Training full model")
        name = basename + '_full'
        full_model = linear.LinearClassifier(best_alpha, penalty=penalty, fit_intercept=intercept, output_dir=output_dir, name=name, pos_label=pos_label, lower=lower)
        full_model.fit(X, Y, train_weights=weights, col_names=col_names)
        if save_model:
            full_model.save()
    return full_model





def prepare_data(X, Y, weights=None, predictions=None, loss='log', normalize=True):
    """
    Expand the feature matrix and label matrix by converting items with multiple labels to multiple rows with 1 each
    :param X: feature matrix (n_items, n_features)
    :param Y: label matrix (n_items, n_classes)
    :param weights: (n_items, )
    :param predictions: optional vector of predcitions to expand in parallel
    :param loss: loss function (determines whether to expand or average
    :param normalize: convert labels into a weighted combination that adds to one
    :return: 
    """
    n_items, n_classes = Y.shape
    pred_return = None
    if loss == 'log':
        # duplicate and down-weight items with multiple labels
        X_list = []
        Y_list = []
        weights_list = []
        pred_list = []
        if weights is None:
            weights = np.ones(n_items)
        # process each item
        for i in range(n_items):
            labels = Y[i, :]
            # sum the total number of annotations given to this item
            total = float(labels.sum())
            # duplicate items with all labels and weights
            for index, count in enumerate(labels):
                # if there is at least one annotation for this class
                if count > 0:
                    # create a row representing an annotation for this class
                    X_list.append(X[i, :])
                    label_vector = np.zeros(n_classes, dtype=int)
                    label_vector[index] = 1
                    Y_list.append(label_vector)
                    # give it a weight based on prior weighting and the proportion of annotations for this class
                    if normalize:
                        weights_list.append(weights[i] * count/total)
                    else:
                        weights_list.append(weights[i] * count)
                    # also append the prediction if given
                    if predictions is not None:
                        pred_list.append(predictions[i])

        # concatenate the newly form lists
        if sparse.issparse(X):
            X_return = sparse.vstack(X_list)
        else:
            X_return = np.vstack(X_list)
        Y_return = np.array(Y_list)
        weights_return = np.array(weights_list)
        if predictions is not None:
            pred_return = np.array(pred_list)
    else:
        sys.exit("Loss %s not recognized" % loss)

    if predictions is None:
        return X_return, Y_return, weights_return
    else:
        return X_return, Y_return, weights_return, pred_return


if __name__ == '__main__':
    main()
