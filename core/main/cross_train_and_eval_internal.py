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
    usage = "%prog project_dir subset cross_field_name config.json "
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='calib_prop', default=0.33,
                      help='Percent to use for the calibration part of each split: default=%default')
    parser.add_option('-t', dest='nontest_prop', default=1.0,
                      help='Proportion of non-test data to use: default=%default')
    parser.add_option('--prefix', dest='prefix', default=None,
                      help='Prefix to _subset_fieldname: default=%default')
    parser.add_option('--max_folds', dest='max_folds', default=None,
                      help='Limit the number of partitions to test: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|BLR]: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='Covariate shift method [None|LR]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--repeats', dest='repeats', default=1,
                      help='Number of repeats with random calibration/test splits: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')
    parser.add_option('--verbose', action="store_true", dest="verbose", default=False,
                      help='Print more output: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    field_name = args[2]
    config_file = args[3]

    calib_prop = float(options.calib_prop)
    nontest_prop = float(options.nontest_prop)
    prefix = options.prefix
    max_folds = options.max_folds
    if max_folds is not None:
        max_folds = int(max_folds)
    #sampling = options.sampling
    model_type = options.model
    label = options.label
    penalty = options.penalty
    cshift = options.cshift
    objective = options.objective
    intercept = not options.no_intercept
    n_dev_folds = int(options.n_dev_folds)
    repeats = int(options.repeats)
    if options.seed is not None:
        np.random.seed(int(options.seed))
    verbose = options.verbose

    pos_label = 1
    average = 'micro'

    cross_train_and_eval(project_dir, subset, field_name, config_file, calib_prop, nontest_prop, prefix, max_folds, model_type, label, penalty, cshift, intercept, n_dev_folds, repeats, verbose, pos_label, average, objective)


def cross_train_and_eval(project_dir, subset, field_name, config_file, calib_prop=0.33, nontest_prop=1.0, prefix=None, max_folds=None, model_type='LR', label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro', objective='f1'):

    model_basename = subset + '_' + field_name
    if prefix is not None:
        model_basename = prefix + '_' + model_basename

    logfile = os.path.join(dirs.dir_logs(project_dir), model_basename + '.json')
    fh.makedirs(dirs.dir_logs(project_dir))
    log = {
        'project': project_dir,
        'subset': subset,
        'field_name': field_name,
        'config_file': config_file,
        'calib_prop': calib_prop,
        'train_prop': nontest_prop,
        'prefix': prefix,
        'max_folds': max_folds,
        'model_type': model_type,
        'label': label,
        'penalty': penalty,
        'cshift': cshift,
        'intercept': intercept,
        'objective': objective,
        'n_dev_folds': n_dev_folds,
        'repeats': repeats,
        'pos_label': pos_label,
        'average': average
    }
    fh.write_to_json(log, logfile)

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

    if max_folds is None:
        max_folds = len(field_vals)

    for v_i, v in enumerate(field_vals[:max_folds]):

        print("\nTesting on %s" % v)
        nontest_selector = metadata[field_name] != v
        nontest_subset = metadata[nontest_selector]
        nontest_items = list(nontest_subset.index)
        n_nontest = len(nontest_items)

        test_selector = metadata[field_name] == v
        test_subset = metadata[test_selector]
        test_items = test_subset.index.tolist()
        n_test = len(test_items)

        # load all labels
        label_dir = dirs.dir_labels(project_dir, subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
        n_items, n_classes = labels_df.shape

        # subsample the non-test items if desired
        if nontest_prop < 1.0:
            np.random.shuffle(nontest_items)
            nontest_items = np.random.choice(nontest_items, size=int(n_nontest * nontest_prop), replace=False)
            n_nontest = len(nontest_items)

        nontest_labels = labels_df.loc[nontest_items]

        if cshift is not None:
            print("Training a classifier for covariate shift")
            # start by learning to discriminate test from non-test data
            train_test_labels = np.zeros((n_items, 2), dtype=int)
            train_test_labels[nontest_selector, 0] = 1
            train_test_labels[test_selector, 1] = 1
            train_test_labels_df = pd.DataFrame(train_test_labels, index=labels_df.index, columns=[0, 1])
            model_name = model_basename + '_' + str(v) + '_' + 'cshift'
            model, dev_f1, dev_cal, _, _ = train.train_model_with_labels(project_dir, model_type, model_name, subset, train_test_labels_df, feature_defs, penalty=penalty, intercept=intercept, n_dev_folds=n_dev_folds, verbose=False)

            train_test_pred_df, train_test_probs_df = predict.predict(project_dir, model, model_name, subset, label, verbose=verbose)
            print("Min: %0.4f" % train_test_probs_df[1].min())
            print("Max: %0.4f" % train_test_probs_df[1].max())
            # base the weights on the probability of each item being a training item
            weights = n_nontest / float(n_test) * (1.0/train_test_probs_df[0].values - 1)
            print("Min weight: %0.4f" % weights[nontest_selector].min())
            print("Ave weight: %0.4f" % weights[nontest_selector].mean())
            print("Max weight: %0.4f" % weights[nontest_selector].max())
            print("Min weight: %0.4f" % weights.min())
            print("Ave weight: %0.4f" % weights.mean())
            print("Max weight: %0.4f" % weights.max())
            weights_df = pd.DataFrame(weights, index=labels_df.index)
        else:
            weights_df = None

        # repeat the following process multiple times with different random splits of calibration / test data
        for r in range(repeats):
            output_df = pd.DataFrame([], columns=['N', 'estimate', 'RMSE', '95lcl', '95ucl', 'contains_test'])

            model_name = model_basename + '_' + str(v) + '_' + str(r)

            # split the non-test items into train and calibration
            n_calib = int(n_nontest * calib_prop)
            np.random.shuffle(nontest_items)
            calib_items = nontest_items[:n_calib]
            train_items = nontest_items[n_calib:]

            train_labels = labels_df.loc[train_items]
            calib_labels = labels_df.loc[calib_items]
            test_labels = labels_df.loc[test_items]

            # get the label proportions from the test and non-test data
            test_props, test_estimate, test_std = get_estimate_and_std(test_labels)
            output_df.loc['test'] = [n_test, test_estimate, 0, test_estimate - 2 * test_std, test_estimate + 2 * test_std, 1]

            nontest_props, nontest_estimate, nontest_std = get_estimate_and_std(nontest_labels)
            nontest_rmse = np.sqrt((nontest_estimate - test_estimate)**2)
            nontest_contains_test = test_estimate > nontest_estimate - 2 * nontest_std and test_estimate < nontest_estimate + 2 * nontest_std
            output_df.loc['nontest'] = [n_nontest, nontest_estimate, nontest_rmse, nontest_estimate - 2 * nontest_std, nontest_estimate + 2 * nontest_std, nontest_contains_test]

            # train a model
            print("Doing training")
            model, dev_f1, dev_cal, acc_cfm, pvc_cfm = train.train_model_with_labels(project_dir, model_type, model_name, subset, labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items, penalty=penalty, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, verbose=verbose)

            # predict on the calibration and test sets
            print("Doing prediction on calibration items")
            calib_predictions, calib_pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=calib_items, verbose=verbose)

            print("Doing prediction on test items")
            test_predictions, test_pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose)

            # evaluate the model on the calibration and test data
            print("Doing evaluation")
            f1_cal, acc_cal = evaluate_predictions.evaluate_predictions(calib_labels, calib_predictions, pos_label=pos_label, average=average)
            f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels, test_predictions, pos_label=pos_label, average=average)
            results_df = pd.DataFrame([], columns=['f1', 'acc'])
            results_df.loc['calibration'] = [f1_cal, acc_cal]
            results_df.loc['test'] = [f1_test, acc_test]
            results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))

            # first check results without any correction
            # average the preditions (assuming binary labels)
            cc_estimate = np.mean(test_predictions[label].values)
            cc_rmse = np.sqrt((cc_estimate - test_estimate)**2)

            # average the predicted probabilities for the positive label (assuming binary labels)
            pcc_estimate = np.mean(test_pred_probs[1].values)
            pcc_rmse = np.sqrt((pcc_estimate - test_estimate)**2)

            output_df.loc['CC'] = [n_test, cc_estimate, cc_rmse, 0, 1, np.nan]
            output_df.loc['PCC'] = [n_test, pcc_estimate, pcc_rmse, 0, 1, np.nan]

            # do the two basic corrections, based on the calibration data
            print("ACC internal")
            calib_labels_expanded, calib_weights_expanded, calib_predictions_expanded = expand_labels(calib_labels.values, calib_predictions.values)
            acc = calibration.compute_acc(calib_labels_expanded, calib_predictions_expanded, n_classes, calib_weights_expanded)
            acc_corrected = calibration.apply_acc_binary(test_predictions.values, acc)
            acc_estimate = acc_corrected[1]
            acc_rmse = np.sqrt((acc_estimate - test_estimate) ** 2)
            output_df.loc['ACC_int'] = [n_calib, acc_estimate, acc_rmse, 0, 1, np.nan]

            print("PVC internal")
            pvc = calibration.compute_pvc(calib_labels_expanded, calib_predictions_expanded, n_classes, weights=calib_weights_expanded)
            pvc_corrected = calibration.apply_pvc(test_predictions.values, pvc)
            pvc_estimate = pvc_corrected[1]
            pvc_rmse = np.sqrt((pvc_estimate - test_estimate) ** 2)
            output_df.loc['PVC_int'] = [n_calib, pvc_estimate, pvc_rmse, 0, 1, np.nan]

            # do IVAP for calibration
            print("Venn")
            test_pred_ranges = ivap.estimate_probs_from_labels(project_dir, model, model_name, subset, subset, labels_df, calib_items, test_items, weights_df=weights_df)
            combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])

            pred_range = np.mean(test_pred_ranges, axis=0)
            venn_estimate = np.mean(combo)
            venn_rmse = np.sqrt((venn_estimate - test_estimate)**2)
            venn_contains_test = pred_range[0] < test_estimate < pred_range[1]
            output_df.loc['Venn'] = [n_calib, venn_estimate, venn_rmse, pred_range[0], pred_range[1], venn_contains_test]

            output_filename = os.path.join(dirs.dir_models(project_dir), model_name, field_name + '_' + str(v) + '.csv')
            output_df.to_csv(output_filename)


def expand_labels(labels, other=None):
    weights = labels.sum(axis=1)
    labels_list = []
    weights_list = []
    other_list = []
    n_items, n_classes = labels.shape
    for c in range(n_classes):
        c_max = labels[:, c].max()
        for i in range(c_max):
            items = np.array(labels[:, c] > i)
            labels_list.append(np.ones(np.sum(items), dtype=int) * c)
            weights_list.append(weights[items])
            if other is not None:
                if other.ndim == 1:
                    other_list.append(other[items])
                else:
                    other_list.append(other[items, :])
    labels = np.hstack(labels_list)
    weights = np.hstack(weights_list)
    if other is None:
        return labels, weights
    else:
        if other.ndim == 1:
            return labels, weights, np.hstack(other_list)
        else:
            return labels, weights, np.vstack(other_list)


def get_estimate_and_std(labels_df):
    n_items, n_classes = labels_df.shape
    assert n_classes == 2
    labels = labels_df.values.copy()

    labels = labels / np.reshape(labels.sum(axis=1), (len(labels), 1))
    props = np.mean(labels, axis=0)
    estimate = props[1]
    std = np.sqrt(estimate * (1 - estimate) / float(n_items))
    return props, estimate, std


if __name__ == '__main__':
    main()
