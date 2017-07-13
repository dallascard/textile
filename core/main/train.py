import os
import sys
from optparse import OptionParser

import numpy as np
from sklearn.model_selection import KFold

from ..util import file_handling as fh
from ..models import lr, blr, evaluation
from ..preprocessing import features
from ..util import dirs


def main():
    usage = "%prog project_dir subset model_name config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|BLR]: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--weights', dest='weights_file', default=None,
                      help='Weights file: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    #parser.add_option('--objective', dest='objective', default='f1',
    #                  help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_classes', dest='n_classes', default=None,
                      help='Specify the number of classes (None=max[training labels]): default=%default')
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
    label = options.label
    weights_file = options.weights_file
    penalty = options.penalty
    #objective = options.objective
    intercept = not options.no_intercept
    n_classes = options.n_classes
    if n_classes is not None:
        n_classes = int(n_classes)
    n_dev_folds = int(options.n_dev_folds)
    if options.seed is not None:
        np.random.seed(int(options.seed))

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    #feature_defs = []
    #for feature_string in args[4:]:
    #    feature_defs.append(features.parse_feature_string(feature_string))

    #feature_def1 = features.FeatureDef('words', transform='binarize')
    #feature_def2 = features.FeatureDef('bigrams', min_df=3)
    #feature_defs = [feature_def1, feature_def2]
    #feature_defs = [feature_def1]

    train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file, n_classes=n_classes, penalty=penalty, intercept=intercept, n_dev_folds=n_dev_folds)


def train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file=None, items_to_use=None,
                n_classes=2, penalty='l2', alpha_min=0.01, alpha_max=1000, n_alphas=8, intercept=True,
                n_dev_folds=5, save_model=True):

    label_dir = dirs.dir_labels(project_dir, subset)
    features_dir = dirs.dir_features(project_dir, subset)

    labels = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
    n_items = len(labels)

    indices_to_use = None
    if items_to_use is not None:
        item_index = dict(zip(labels.index, range(n_items)))
        indices_to_use = [item_index[i] for i in items_to_use]
        labels = labels.loc[items_to_use]
        n_items = len(labels)

    print("loading features")
    feature_list = []
    feature_signatures = []
    for feature_def in feature_defs:
        print(feature_def)
        name = feature_def.name
        feature = features.load_from_file(input_dir=features_dir, basename=name)
        # take a subset of the rows, if requested
        print("Initial shape = (%d, %d)" % feature.get_shape())
        if indices_to_use is not None:
            print("Taking subset of items")
            feature = features.create_from_feature(feature, indices_to_use)
            print("New shape = (%d, %d)" % feature.get_shape())
        feature.threshold(feature_def.min_df)
        feature.transform(feature_def.transform)
        print("Final shape = (%d, %d)" % feature.get_shape())
        feature_list.append(feature)
        if save_model:
            feature_signatures.append(features.get_feature_signature(feature_def, feature))

    if save_model:
        output_dir = os.path.join(dirs.dir_models(project_dir), model_name)
        fh.makedirs(output_dir)
        fh.write_to_json(feature_signatures, os.path.join(output_dir, 'features.json'), sort_keys=False)

    features_concat = features.concatenate(feature_list)
    col_names = features_concat.terms

    X = features_concat.get_counts().tocsr()
    y = labels[label].as_matrix()

    weights = None
    weights_k = None
    if weights_file is not None:
        weights_df = fh.read_csv_to_df(weights_file)
        assert np.all(weights_df.index == labels.index)
        weights = weights_df['weight'].values

    if n_classes is None:
        n_classes = int(np.max(y))+1
    bincount = np.bincount(y, minlength=n_classes)
    print("Using %d classes" % n_classes)
    train_proportions = bincount / float(bincount.sum())
    print("Train proportions: %s" % str(train_proportions.tolist()))

    print("Train feature matrix shape: (%d, %d)" % X.shape)

    try:
        assert np.array(features_concat.items == labels.index).all()
    except AssertionError:
        print("mismatch in items between labels and features")
        print(features_concat.items[:5])
        print(labels.index[:5])
        sys.exit()

    kfold = KFold(n_splits=n_dev_folds, shuffle=True)
    #kfold = KFold(len(y), n_folds=n_dev_folds, shuffle=True)
    if n_alphas > 1:
        alpha_factor = np.power(alpha_max / alpha_min, 1.0/(n_alphas-1))
        alphas = np.array(alpha_min * np.power(alpha_factor, np.arange(n_alphas)))
    else:
        alphas = [alpha_min]

    mean_train_f1s = np.zeros(n_alphas)
    mean_dev_f1s = np.zeros(n_alphas)
    mean_model_size = np.zeros(n_alphas)

    print("%s\t%s\t%s\t%s\t%s" % ('iter', 'alpha', 'size', 'f1_trn', 'f1_dev'))

    if model_type == 'LR':
        for alpha_i, alpha in enumerate(alphas):
            model = lr.LR(alpha, penalty=penalty, fit_intercept=intercept, n_classes=n_classes)

            #for i, (train, dev) in enumerate(kfold):
            for train_indices, dev_indices in kfold.split(X):
                if weights is not None:
                    weights_k = weights[train_indices]
                model.fit(X[train_indices, :], y[train_indices], col_names, sample_weights=weights_k)

                train_predictions = model.predict(X[train_indices, :])
                dev_predictions = model.predict(X[dev_indices, :])
                dev_probs = model.predict_probs(X[dev_indices, :])

                train_f1 = evaluation.f1_score(y[train_indices], train_predictions, n_classes)
                dev_f1 = evaluation.f1_score(y[dev_indices], dev_predictions, n_classes)
                #dev_acc = evaluation.acc_score(y[dev], dev_prediction)
                #dev_cal = evaluation.evaluate_calibration_mse(y[dev], dev_probs)

                mean_train_f1s[alpha_i] += train_f1 / float(n_dev_folds)
                mean_dev_f1s[alpha_i] += dev_f1 / float(n_dev_folds)
                #mean_dev_accs[alpha_i] += dev_acc / float(n_dev_folds)
                #mean_dev_cals[alpha_i] += dev_cal / float(n_dev_folds)

                mean_model_size[alpha_i] += model.get_model_size() / float(n_dev_folds)

            print("%d\t%0.2f\t%.1f\t%0.3f\t%0.3f" % (alpha_i, alpha, mean_model_size[alpha_i], mean_train_f1s[alpha_i], mean_dev_f1s[alpha_i]))

            #acc_cfms.append(np.mean(alpha_acc_cfms, axis=0))
            #pacc_cfms.append(np.mean(alpha_pacc_cfms, axis=0))
            #pvc_cfms.append(np.mean(alpha_pvc_cfms, axis=0))

        best_f1_alpha_index = np.argmax(mean_dev_f1s)
        #best_acc_alpha_index = np.argmax(mean_dev_accs)
        #best_cal_alpha_index = np.argmin(mean_dev_cals)

        best_f1_alpha = alphas[best_f1_alpha_index]
        #best_acc_alpha = alphas[best_acc_alpha_index]
        #best_cal_alpha = alphas[best_cal_alpha_index]

        #best_acc_cfm = acc_cfms[best_acc_alpha_index]
        #best_pacc_cfm = pacc_cfms[best_acc_alpha_index]
        #best_pvc_cfm = pvc_cfms[best_acc_alpha_index]

        print("Best: alpha = %.3f, dev f1 = %.3f" % (best_f1_alpha, np.max(mean_dev_f1s)))
        #print "Best acc alpha = %.3f" % best_acc_alpha
        #print "Best cal alpha = %.3f" % best_cal_alpha

        print("Training full model")
        model = lr.LR(best_f1_alpha, penalty=penalty, fit_intercept=intercept, n_classes=n_classes)
        model.fit(X, y, col_names)

    elif model_type == 'BLR':
        print("Fitting single model with ARD")
        model = blr.BLR(alpha=None, fit_intercept=intercept, n_classes=n_classes)
        model.fit(np.array(X.todense()), y, col_names, sample_weights=weights, batch=True, multilevel=True, ard=True)
    else:
        sys.exit("Model type not recognized")

    output_dir = os.path.join(dirs.dir_models(project_dir), model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(output_dir)

    return model


if __name__ == '__main__':
    main()
