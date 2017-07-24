import os
from optparse import OptionParser

import numpy as np
import pandas as pd

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

    for v_i, v in enumerate(field_vals[2:3]):
        model_name = model_type + '_' + str(v)

        output_df = pd.DataFrame([], columns=['N', 'estimate', 'RMSE', '95lcl', '95ucl', 'contains_test'])

        print("\nTesting on %s" % v)
        train_subset = metadata[metadata[field_name] != v]
        train_items = train_subset.index
        n_train = len(train_items)
        non_train_subset = metadata[metadata[field_name] == v]
        non_train_items = non_train_subset.index.tolist()

        n_non_train = len(non_train_items)
        n_calib = int(calib_prop * n_non_train)
        np.random.shuffle(non_train_items)
        calib_items = non_train_items[:n_calib]
        test_items = non_train_items[n_calib:]
        n_test = len(test_items)

        # load all labels
        label_dir = dirs.dir_labels(project_dir, subset)
        labels = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
        if n_classes is None:
            n_classes = int(np.max(labels)) + 1
            print("Assuming %d classes" % n_classes)
        train_labels = labels.loc[train_items]
        calib_labels = labels.loc[calib_items]
        test_labels = labels.loc[test_items]

        print("Size of test set = %d" % int(n_non_train - n_calib))
        test_props, test_estimate, test_std = get_estimate_and_std(test_labels, n_classes)
        output_df.loc['test'] = [n_test, test_estimate, 0, test_estimate - 2 * test_std, test_estimate + 2 * test_std, 1]

        train_props, train_estimate, train_std = get_estimate_and_std(train_labels, n_classes)
        train_rmse = np.sqrt((train_estimate - test_estimate)**2)
        train_contains_test = test_estimate > train_estimate - 2 * train_std and test_estimate < train_estimate + 2 * train_std
        output_df.loc['train'] = [n_train, train_estimate, train_rmse, train_estimate - 2 * train_std, train_estimate + 2 * train_std, train_contains_test]

        calib_props, calib_estimate, calib_std = get_estimate_and_std(calib_labels, n_classes)
        calib_rmse = np.sqrt((calib_estimate - test_estimate)**2)
        calib_contains_test = test_estimate > calib_estimate - 2 * calib_std and calib_estimate < calib_estimate + 2 * calib_std
        output_df.loc['calibration'] = [n_calib, calib_estimate, calib_rmse, calib_estimate - 2 * calib_std, calib_estimate + 2 * calib_std, calib_contains_test]

        print("Doing training")
        model = train.train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file, items_to_use=train_items, n_classes=n_classes, penalty=penalty, intercept=intercept, n_dev_folds=n_dev_folds, verbose=False)

        print("Doing prediction")
        calib_predictions, calib_pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=calib_items)
        test_predictions, test_pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items)

        print("Doing evaluation")
        f1, acc = evaluate_predictions.evaluate_predictions(test_labels, test_predictions, n_classes=n_classes, pos_label=pos_label, average=average)
        results_df = pd.DataFrame([f1, acc], columns=['f1', 'acc'])
        results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))

        cc_estimate = np.sum(test_predictions) / float(n_test)
        cc_rmse = np.sqrt((cc_estimate - test_estimate)**2)
        pcc_estimate = np.mean(test_pred_probs, axis=1)[1]
        pcc_rmse = np.sqrt((pcc_estimate - test_estimate)**2)

        output_df.loc['CC'] = [n_test, cc_estimate, cc_rmse, 0, 1, np.nan]
        output_df.loc['PCC'] = [n_test, pcc_estimate, pcc_rmse, 0, 1, np.nan]

        # do some sort of calibration here (ACC, PACC, PVC)
        print("ACC correction")
        acc = calibration.compute_acc(calib_labels.values, calib_predictions.values, n_classes)
        acc_corrected = calibration.apply_acc_binary(test_predictions.values, acc)
        acc_estimate = acc_corrected[1]
        acc_rmse = np.sqrt((acc_estimate - test_estimate) ** 2)
        output_df.loc['ACC'] = [n_calib, acc_estimate, acc_rmse, 0, 1, np.nan]

        print("PVC correction")
        pvc = calibration.compute_pvc(calib_labels.values, calib_predictions.values, n_classes)
        pvc_corrected = calibration.apply_pvc(test_predictions.values, pvc)
        pvc_estimate = pvc_corrected
        pvc_rmse = np.sqrt((pvc_estimate - test_estimate) ** 2)
        output_df.loc['PVC'] = [n_calib, pvc_estimate, pvc_rmse, 0, 1, np.nan]

        test_pred_ranges = ivap.estimate_probs_brute_force(project_dir, model, model_name, subset, subset, label, calib_items, test_items)
        combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])
        test_label_list = test_labels[label]
        pred_prob_list = test_pred_probs[1]
        #for i in range(len(test_label_list)):
        #    print(i, test_label_list[i], pred_prob_list[i], test_pred_ranges[i, :], combo[i])

        pred_range = np.mean(test_pred_ranges, axis=0)
        venn_estimate = np.mean(combo)
        venn_rmse = np.sqrt((venn_estimate - test_estimate)**2)
        venn_contains_test = test_estimate > pred_range[0] and calib_estimate < pred_range[1]
        output_df.loc['Venn'] = [n_calib, venn_estimate, venn_rmse, pred_range[0], pred_range[1], venn_contains_test]

        output_filename = os.path.join(dirs.dir_models(project_dir), model_name, field_name + '_' + str(v) + '.csv')
        output_df.to_csv(output_filename)


def get_estimate_and_std(subset_labels, n_classes):
    props = evaluation.compute_proportions(subset_labels, n_classes)
    estimate = props[1]
    std = np.sqrt(estimate * (1 - estimate) / float(len(subset_labels)))
    return props, estimate, std



if __name__ == '__main__':
    main()
