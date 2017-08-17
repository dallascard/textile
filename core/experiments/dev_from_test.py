import os
from optparse import OptionParser

import numpy as np
import pandas as pd

from core.util import file_handling as fh
from core.preprocessing import features
from core.main import train, predict, evaluate_predictions
from core.models import calibration, ivap
from core.util import dirs


"""
Run an experiment by using each fold of a partition as test data, and training on the rest, doing cross-validation
on the training set to choose a model, and sampling from the test data to get a calibration set 
"""


def main():
    usage = "%prog project_dir subset cross_field_name config.json "
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='calib_prop', default=0.33,
                      help='Percent to use for the calibration part of each split: default=%default')
    parser.add_option('-t', dest='train_prop', default=0.33,
                      help='Proportion of training data to use: default=%default')
    parser.add_option('--prefix', dest='prefix', default=None,
                      help='Prefix to _subset_fieldname: default=%default')
    parser.add_option('--max_folds', dest='max_folds', default=None,
                      help='Limit the number of partitions to test: default=%default')
    parser.add_option('--min_val', dest='min_val', default=None,
                      help='Minimum value of partition to test on: default=%default')
    parser.add_option('--max_val', dest='max_val', default=None,
                      help='Maximum value of partition to test on: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|MLP]: default=%default')
    parser.add_option('--loss', dest='loss', default='log',
                      help='Loss function [log|brier]: default=%default')
    parser.add_option('--dh', dest='dh', default=0,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
                      help='Minimum value of training hyperparameter: default=%default')
    parser.add_option('--alpha_max', dest='alpha_max', default=1000,
                      help='Maximum value of training hyperparameter: default=%default')
    parser.add_option('--ensemble', action="store_true", dest="ensemble", default=False,
                      help='Make an ensemble from cross-validation, instead of training one model: default=%default')
    parser.add_option('--exclude_calib', action="store_true", dest="exclude_calib", default=False,
                      help='Exclude the calibration data from the evalaution: default=%default')
    parser.add_option('--calib_pred', action="store_true", dest="calib_pred", default=False,
                      help='Use predictions on calibration items, rather than given labels: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='Covariate shift method [None]: default=%default')
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
    train_prop = float(options.train_prop)
    prefix = options.prefix
    max_folds = options.max_folds
    min_val = options.min_val
    max_val = options.max_val
    if max_folds is not None:
        max_folds = int(max_folds)
    model_type = options.model
    loss = options.loss
    dh = int(options.dh)
    alpha_min = float(options.alpha_min)
    alpha_max = float(options.alpha_max)
    do_ensemble = options.ensemble
    exclude_calib = options.exclude_calib
    calib_pred = options.calib_pred
    label = options.label
    penalty = options.penalty
    cshift = options.cshift
    objective = options.objective
    intercept = not options.no_intercept
    n_dev_folds = int(options.n_dev_folds)
    repeats = int(options.repeats)
    seed = options.seed
    if options.seed is not None:
        seed = int(seed)
        np.random.seed(seed)
    verbose = options.verbose

    pos_label = 1
    average = 'micro'

    cross_train_and_eval(project_dir, subset, field_name, config_file, calib_prop, train_prop, prefix, max_folds, min_val, max_val, model_type, loss, do_ensemble, dh, label, penalty, cshift, intercept, n_dev_folds, repeats, verbose, pos_label, average, objective, seed, calib_pred, exclude_calib, alpha_min, alpha_max)


def cross_train_and_eval(project_dir, subset, field_name, config_file, calib_prop=0.33, train_prop=1.0, prefix=None, max_folds=None, min_val=None, max_val=None, model_type='LR', loss='log', do_ensemble=False, dh=0, label='label', penalty='l2', cshift=None, intercept=True, n_dev_folds=5, repeats=1, verbose=False, pos_label=1, average='micro', objective='f1', seed=None, use_calib_pred=False, exclude_calib=False, alpha_min=0.01, alpha_max=1000):

    model_basename = subset + '_' + field_name
    if prefix is not None:
        model_basename = prefix + '_' + model_basename

    # save the experiment parameters to a log file
    logfile = os.path.join(dirs.dir_logs(project_dir), model_basename + '.json')
    fh.makedirs(dirs.dir_logs(project_dir))
    log = {
        'project': project_dir,
        'subset': subset,
        'field_name': field_name,
        'config_file': config_file,
        'calib_prop': calib_prop,
        'train_prop': train_prop,
        'prefix': prefix,
        'max_folds': max_folds,
        'model_type': model_type,
        'loss': loss,
        'dh': dh,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'do_ensemble': do_ensemble,
        'label': label,
        'penalty': penalty,
        'cshift': cshift,
        'intercept': intercept,
        'objective': objective,
        'n_dev_folds': n_dev_folds,
        'repeats': repeats,
        'pos_label': pos_label,
        'average': average,
        'use_calib_pred': use_calib_pred,
        'exclude_calib': exclude_calib
    }
    fh.write_to_json(log, logfile)

    # load the features specified in the config file
    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    # load the file that contains metadata about each item
    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata[field_name].values))
    field_vals.sort()
    print(field_vals)

    # exclude certain values of the partition if desired
    if min_val is not None:
        field_vals = [v for v in field_vals if v >= float(min_val)]

    if max_val is not None:
        field_vals = [v for v in field_vals if v <= float(max_val)]

    if max_folds is None:
        max_folds = len(field_vals)

    # repeat the following value for each fold of the partition of interest (up to max_folds, if given)
    for v_i, v in enumerate(field_vals[:max_folds]):
        print("\nTesting on %s" % v)
        # first, split into training and non-train data based on the field of interest
        train_selector = metadata[field_name] != v
        train_subset = metadata[train_selector]
        train_items = list(train_subset.index)
        n_train = len(train_items)

        non_train_selector = metadata[field_name] == v
        non_train_subset = metadata[non_train_selector]
        non_train_items = non_train_subset.index.tolist()
        n_non_train = len(non_train_items)

        print("Train: %d, non-train: %d" % (n_train, n_non_train))

        # load all labels
        label_dir = dirs.dir_labels(project_dir, subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
        n_items, n_classes = labels_df.shape
        train_labels = labels_df.loc[train_items]

        # if desired, attempt to learn weights for the training data using techniques for covariate shift
        # TODO: update this to work with ALL of the data
        if cshift is not None:
            print("Training a classifier for covariate shift")
            # start by learning to discriminate train from non-train data
            train_test_labels = np.zeros((n_items, 2), dtype=int)
            train_test_labels[train_selector, 0] = 1
            train_test_labels[non_train_selector, 1] = 1
            train_test_labels_df = pd.DataFrame(train_test_labels, index=labels_df.index, columns=[0, 1])
            # create a cshift model using the same specifiction as our model below (e.g. LR/MLP, etc.)
            model_name = model_basename + '_' + str(v) + '_' + 'cshift'
            model, dev_f1, dev_acc, dev_cal, _, _ = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, train_test_labels_df, feature_defs, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, intercept=intercept, n_dev_folds=n_dev_folds, save_model=True, do_ensemble=do_ensemble, dh=dh, seed=seed, verbose=False)
            print("cshift results: %0.4f f1, %0.4f acc" % (dev_f1, dev_acc))

            # take predictions from model on the training data
            train_test_pred_df, train_test_probs_df = predict.predict(project_dir, model, model_name, subset, label, verbose=verbose)
            # display the min and max probs
            print("Min: %0.4f" % train_test_probs_df[1].min())
            print("Max: %0.4f" % train_test_probs_df[1].max())
            # use the estimated probability of each item being a training item to compute item weights
            weights = n_train / float(n_non_train) * (1.0/train_test_probs_df[0].values - 1)
            # print a summary of the weights from just the training items
            print("Min weight: %0.4f" % weights[train_selector].min())
            print("Ave weight: %0.4f" % weights[train_selector].mean())
            print("Max weight: %0.4f" % weights[train_selector].max())
            # print a summary of all weights
            print("Min weight: %0.4f" % weights.min())
            print("Ave weight: %0.4f" % weights.mean())
            print("Max weight: %0.4f" % weights.max())
            # create a data frame with this information
            weights_df = pd.DataFrame(weights, index=labels_df.index)
        else:
            weights_df = None

        # repeat the following process multiple times with different random splits of train / calibration / test data
        for r in range(repeats):

            # next, take a random subset of the training data (and ignore the rest), to simulate fewer annotated items
            if train_prop < 1.0:
                np.random.shuffle(train_items)
                train_items_r = np.random.choice(train_items, size=int(n_train * train_prop), replace=False)
                n_train_r = len(train_items_r)

            # create a data frame to hold a summary of the results
            output_df = pd.DataFrame([], columns=['N', 'estimate', 'RMSE', '95lcl', '95ucl', 'contains_test'])
            # create a unique name ofr this model
            model_name = model_basename + '_' + str(v) + '_' + str(r)

            # now, divide the non-train data into a calibration and a test set
            n_calib = int(calib_prop * n_non_train)
            np.random.shuffle(non_train_items)
            calib_items = non_train_items[:n_calib]
            test_items = non_train_items[n_calib:]
            n_test = len(test_items)

            print("%d %d %d" % (n_train_r, n_calib, n_test))
            train_labels_r_df = labels_df.loc[train_items_r]
            calib_labels_df = labels_df.loc[calib_items]
            test_labels_df = labels_df.loc[test_items]
            non_train_labels_df = labels_df.loc[non_train_items]

            # get the true proportion of labels in the test OR non-training data (calibration and test combined)
            if exclude_calib:
                test_props, test_estimate, test_std = get_estimate_and_std(test_labels_df)
            else:
                test_props, test_estimate, test_std = get_estimate_and_std(non_train_labels_df)
            output_df.loc['test'] = [n_test, test_estimate, 0, test_estimate - 2 * test_std, test_estimate + 2 * test_std, 1]

            # get the same estimate from training data
            train_props, train_estimate, train_std = get_estimate_and_std(train_labels_r_df)
            # compute the error of this estimate
            train_rmse = np.sqrt((train_estimate - test_estimate)**2)
            train_contains_test = test_estimate > train_estimate - 2 * train_std and test_estimate < train_estimate + 2 * train_std
            output_df.loc['train'] = [n_train_r, train_estimate, train_rmse, train_estimate - 2 * train_std, train_estimate + 2 * train_std, train_contains_test]

            # repeat for calibration data
            calib_props, calib_estimate, calib_std = get_estimate_and_std(calib_labels_df)
            calib_rmse = np.sqrt((calib_estimate - test_estimate)**2)
            # check if the test estimate is within 2 standard deviations of the estimate
            calib_contains_test = test_estimate > calib_estimate - 2 * calib_std and calib_estimate < calib_estimate + 2 * calib_std
            output_df.loc['calibration'] = [n_calib, calib_estimate, calib_rmse, calib_estimate - 2 * calib_std, calib_estimate + 2 * calib_std, calib_contains_test]

            results_df = pd.DataFrame([], columns=['f1', 'acc', 'cal'])

            print("Training model on all labeled data")
            # first train a model on the training and calibration data combined
            calib_and_train_items_r = np.array(list(calib_items) + list(train_items_r))
            model, dev_f1, dev_acc, dev_cal, acc_cfm, pvc_cfm = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, labels_df, feature_defs, weights_df=weights_df, items_to_use=calib_and_train_items_r, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, verbose=verbose)
            results_df.loc['cross_val_all'] = [dev_f1, dev_acc, dev_cal]

            # get labels for test data
            test_predictions_df, test_pred_probs_df = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose)
            f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
            results_df.loc['test_all'] = [f1_test, acc_test, 0.0]

            # combine the predictions on the test and calibration data (unless excluding calibration data from this)
            if exclude_calib:
                test_predictions = test_predictions_df.values
                test_pred_probs = test_pred_probs_df.values
            else:
                # get labels for calibration data
                if use_calib_pred:
                    calib_predictions_df, calib_pred_probs_df = predict.predict(project_dir, model, model_name, subset, label, items_to_use=calib_items, verbose=verbose)
                else:
                    calib_predictions_df = pd.DataFrame(np.argmax(calib_labels_df.values, axis=1), index=calib_labels_df.index)
                    # normalize labels to get (questionable) estimates of probabilities
                    calib_pred_probs_df = pd.DataFrame(calib_labels_df.values / np.array(np.sum(calib_labels_df.values, axis=1).reshape((n_calib, 1)), dtype=float), index=calib_labels_df.index)

                test_predictions = np.r_[test_predictions_df.values, calib_predictions_df.values]
                test_pred_probs = np.vstack([test_pred_probs_df.values, calib_pred_probs_df.values])

            # get the basic error estimates for this model
            cc_estimate = np.mean(test_predictions)
            cc_rmse = np.sqrt((cc_estimate - test_estimate)**2)

            # average the predicted probabilities for the positive label (assuming binary labels)
            pcc_estimate = np.mean(test_pred_probs[:, 1])
            pcc_rmse = np.sqrt((pcc_estimate - test_estimate)**2)

            output_df.loc['CC_all'] = [n_test, cc_estimate, cc_rmse, 0, 1, np.nan]
            output_df.loc['PCC_all'] = [n_test, pcc_estimate, pcc_rmse, 0, 1, np.nan]

            # Now repeat for a model trained on the training data, saving the calibration data for calibration
            print("Training model on training data only")
            model, dev_f1, dev_acc, dev_cal, acc_cfm, pvc_cfm = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items_r, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max,  intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, verbose=verbose)
            results_df.loc['cross_val'] = [dev_f1, dev_acc, dev_cal]

            # predict on calibration data
            calib_predictions_df, calib_pred_probs_df = predict.predict(project_dir, model, model_name, subset, label, items_to_use=calib_items, verbose=verbose)
            f1_cal, acc_cal = evaluate_predictions.evaluate_predictions(calib_labels_df, calib_predictions_df, calib_pred_probs_df, pos_label=pos_label, average=average)
            results_df.loc['calibration'] = [f1_cal, acc_cal, calib_rmse]

            # predict on test data
            test_predictions_df, test_pred_probs_df = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose)
            f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
            results_df.loc['test'] = [f1_test, acc_test, 0.0]
            results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))

            # combine the predictions on the test and calibration data (unless excluding calibration data from this)
            if exclude_calib:
                test_predictions = test_predictions_df.values
                test_pred_probs = test_pred_probs_df.values
            else:
                if not use_calib_pred:
                    calib_predictions_df = pd.DataFrame(np.argmax(calib_labels_df.values, axis=1), index=calib_labels_df.index)
                    # normalize labels to get (questionable) estimates of probabilities
                    calib_pred_probs_df = pd.DataFrame(calib_labels_df.values / np.array(np.sum(calib_labels_df.values, axis=1).reshape((n_calib, 1)), dtype=float), index=calib_labels_df.index)

                test_predictions = np.r_[test_predictions_df.values, calib_predictions_df.values]
                test_pred_probs = np.vstack([test_pred_probs_df.values, calib_pred_probs_df.values])

            # now evaluate in terms of predicted proportions
            # average the predictions (assuming binary labels)
            cc_estimate = np.mean(test_predictions)
            cc_rmse = np.sqrt((cc_estimate - test_estimate)**2)

            # average the predicted probabilities for the positive label (assuming binary labels)
            pcc_estimate = np.mean(test_pred_probs[:, 1])
            pcc_rmse = np.sqrt((pcc_estimate - test_estimate)**2)

            output_df.loc['CC'] = [n_test, cc_estimate, cc_rmse, 0, 1, np.nan]
            output_df.loc['PCC'] = [n_test, pcc_estimate, pcc_rmse, 0, 1, np.nan]

            # expand the data so as to only have singly-labeled, weighted items
            _, calib_labels, calib_weights, calib_predictions = train.prepare_data(np.zeros([n_calib, 2]), calib_labels_df.values, predictions=calib_predictions_df.values)

            # do some sort of calibration here (ACC, PACC, PVC)
            print("ACC correction")
            #calib_labels_expanded, calib_weights_expanded, calib_predictions_expanded = expand_labels(calib_labels.values, calib_predictions.values)
            acc = calibration.compute_acc(calib_labels, calib_predictions, n_classes, weights=calib_weights)
            acc_corrected = calibration.apply_acc_binary(test_predictions, acc)
            acc_estimate = acc_corrected[1]
            acc_rmse = np.sqrt((acc_estimate - test_estimate) ** 2)
            output_df.loc['ACC'] = [n_calib, acc_estimate, acc_rmse, 0, 1, np.nan]

            print("ACC internal")
            acc_corrected = calibration.apply_acc_binary(test_predictions, acc_cfm)
            acc_estimate = acc_corrected[1]
            acc_rmse = np.sqrt((acc_estimate - test_estimate) ** 2)
            output_df.loc['ACC_int'] = [n_calib, acc_estimate, acc_rmse, 0, 1, np.nan]

            print("PVC correction")
            pvc = calibration.compute_pvc(calib_labels, calib_predictions, n_classes, weights=calib_weights)
            pvc_corrected = calibration.apply_pvc(test_predictions, pvc)
            pvc_estimate = pvc_corrected[1]
            pvc_rmse = np.sqrt((pvc_estimate - test_estimate) ** 2)
            output_df.loc['PVC'] = [n_calib, pvc_estimate, pvc_rmse, 0, 1, np.nan]

            print("PVC internal")
            pvc_corrected = calibration.apply_pvc(test_predictions, pvc_cfm)
            pvc_estimate = pvc_corrected[1]
            pvc_rmse = np.sqrt((pvc_estimate - test_estimate) ** 2)
            output_df.loc['PVC_int'] = [n_calib, pvc_estimate, pvc_rmse, 0, 1, np.nan]

            print("Venn")
            test_pred_ranges = ivap.estimate_probs_from_labels(project_dir, model, model_name, subset, subset, labels_df, calib_items, test_items, weights_df=None)
            if not exclude_calib:
                # try also doing IVAP on individual calibration items, using all others
                if use_calib_pred:
                    calib_pred_ranges = []
                    for i in range(n_calib):
                        other_items = calib_items.tolist()
                        other_items.pop(i)
                        calib_pred_ranges.append(ivap.estimate_probs_from_labels(project_dir, model, model_name, subset, subset, labels_df, other_items, calib_items[i], weights_df=None))
                    calib_pred_ranges = np.vstack(calib_pred_ranges)
                else:
                    calib_pred_ranges = np.hstack([calib_labels_df.values[:, 1], calib_labels_df.values[:, 1]])
                test_pred_ranges = np.vstack([test_pred_ranges, calib_pred_ranges])

            combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])

            pred_range = np.mean(test_pred_ranges, axis=0)
            venn_estimate = np.mean(combo)
            venn_rmse = np.sqrt((venn_estimate - test_estimate)**2)
            venn_contains_test = pred_range[0] < test_estimate < pred_range[1]
            output_df.loc['Venn'] = [n_calib, venn_estimate, venn_rmse, pred_range[0], pred_range[1], venn_contains_test]

            output_filename = os.path.join(dirs.dir_models(project_dir), model_name, field_name + '_' + str(v) + '.csv')
            output_df.to_csv(output_filename)


"""
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
"""

def get_estimate_and_std(labels_df):
    n_items, n_classes = labels_df.shape
    assert n_classes == 2
    labels = labels_df.values.copy()

    # normalize the labels across classes
    labels = labels / np.reshape(labels.sum(axis=1), (len(labels), 1))
    # take the mean
    props = np.mean(labels, axis=0)
    estimate = props[1]
    # estimate the variance by pretending this is a binomial distribution
    std = np.sqrt(estimate * (1 - estimate) / float(n_items))
    return props, estimate, std


if __name__ == '__main__':
    main()
