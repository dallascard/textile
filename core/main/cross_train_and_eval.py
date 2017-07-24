import os
import sys
from optparse import OptionParser

import numpy as np

from ..util import file_handling as fh
from ..preprocessing import features
from ..main import train, predict, evaluate_predictions, estimate_proportions
from ..models import evaluation, calibration, ivap
from ..util import dirs


def main():
    usage = "%prog project_dir subset cross_field_name model_basename config.json "
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='calib_prop', default=0.33,
                      help='Percent to use for the calibration part of each split: default=%default')
    #parser.add_option('--sampling', dest='sampling', default='proportional',
    #                  help='How to divide calibration and test data [proportional|random]: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|BLR]: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='Covariate shift method [None]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    #parser.add_option('--objective', dest='objective', default='f1',
    #                  help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_classes', dest='n_classes', default=None,
                      help='Specify the number of classes (None=max(train)+1): default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    field_name = args[2]
    model_basename = args[3]
    config_file = args[4]

    calib_prop = float(options.calib_prop)
    #sampling = options.sampling
    model_type = options.model
    label = options.label
    penalty = options.penalty
    cshift = options.cshift
    #objective = options.objective
    intercept = not options.no_intercept
    n_classes = options.n_classes
    if n_classes is not None:
        n_classes = int(n_classes)
    n_dev_folds = int(options.n_dev_folds)
    if options.seed is not None:
        np.random.seed(int(options.seed))

    pos_label = 1
    average = 'micro'

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    weights_file = None

    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata[field_name].values))
    field_vals.sort()
    print(field_vals)

    for v_i, v in enumerate(field_vals[-3:-2]):
        model_name = model_type + '_' + str(v)

        print("\nTesting on %s" % v)
        train_subset = metadata[metadata[field_name] != v]
        train_items = train_subset.index
        non_train_subset = metadata[metadata[field_name] == v]
        non_train_items = non_train_subset.index.tolist()

        n_non_train = len(non_train_items)
        n_calib = int(calib_prop * n_non_train)
        np.random.shuffle(non_train_items)
        calib_items = non_train_items[:n_calib]
        test_items = non_train_items[n_calib:]

        # load all labels
        label_dir = dirs.dir_labels(project_dir, subset)
        labels = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
        if n_classes is None:
            n_classes = int(np.max(labels)) + 1
            print("Assuming %d classes" % n_classes)
        train_labels = labels.loc[train_items]
        calib_labels = labels.loc[calib_items]
        test_labels = labels.loc[test_items]

        print("Doing training")
        model = train.train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file, items_to_use=train_items, n_classes=n_classes, penalty=penalty, intercept=intercept, n_dev_folds=n_dev_folds, verbose=True)

        print("Doing prediction")
        calib_predictions, calib_pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=calib_items)
        test_predictions, test_pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items)

        print("Doing evaluation")
        evaluate_predictions.evaluate_predictions(test_labels, test_predictions, n_classes=n_classes, pos_label=pos_label, average=average)

        # do some sort of calibration here (ACC, PACC, PVC)
        print("ACC correction")
        acc = calibration.compute_acc(calib_labels.values, calib_predictions.values, n_classes)
        acc_corrected = calibration.apply_acc_binary(test_predictions.values, acc)
        print(acc_corrected)

        print("PVC correction")
        pvc = calibration.compute_pvc(calib_labels.values, calib_predictions.values, n_classes)
        pvc_corrected = calibration.apply_pvc(test_predictions.values, pvc)
        print(pvc_corrected)

        print("Train proportions")
        train_props = evaluation.compute_proportions(train_labels, n_classes)
        print(train_props)

        print("Size of calibration set = %d" % n_calib)
        print("Calibration proportions")
        calib_props = evaluation.compute_proportions(calib_labels, n_classes)
        print(calib_props)

        est_var = calib_props[1] * (1 - calib_props[1]) / float(n_calib)
        beta_a = n_calib * calib_props[0] + 1
        beta_b = n_calib * calib_props[1] + 1
        beta_var = beta_a * beta_b / ((beta_a + beta_b)**2 * (beta_a + beta_b + 1))
        print(np.sqrt(est_var), np.sqrt(beta_var))

        print("Size of test set = %d" % int(n_non_train - n_calib))
        print("Test proportions")
        test_props = evaluation.compute_proportions(test_labels, n_classes)
        print(test_props)

        test_pred_ranges = ivap.estimate_probs_brute_force(project_dir, model, model_name, subset, subset, label, calib_items, test_items)
        combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])
        test_label_list = test_labels[label]
        pred_prob_list = test_pred_probs[1]
        for i in range(len(test_label_list)):
            print(i, test_label_list[i], pred_prob_list[i], test_pred_ranges[i, :], combo[i])

        print("Venn calibration")
        pred_range = np.mean(test_pred_ranges, axis=0)
        print(pred_range)
        print(np.mean(combo))

if __name__ == '__main__':
    main()
