import os
from optparse import OptionParser

import numpy as np
import pandas as pd

from core.util import file_handling as fh
from core.preprocessing import features
from core.main import train, predict, evaluate_predictions
from core.models import calibration, ivap, evaluation
from core.util import dirs





def main():
    usage = "%prog project_dir subset config.json "
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default=500,
                      help='Number of training instances to use: default=%default')
    #parser.add_option('--n_calib', dest='n_calib', default=0,
    #                  help='Number of test instances to use for calibration: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to mdoel name: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|MLP]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
                      help='Minimum value of training hyperparameter: default=%default')
    parser.add_option('--alpha_max', dest='alpha_max', default=1000,
                      help='Maximum value of training hyperparameter: default=%default')
    #parser.add_option('--ensemble', action="store_true", dest="ensemble", default=False,
    #                  help='Make an ensemble from cross-validation, instead of training one model: default=%default')
    #parser.add_option('--exclude_calib', action="store_true", dest="exclude_calib", default=False,
    #                  help='Exclude the calibration data from the evalaution: default=%default')
    #parser.add_option('--calib_pred', action="store_true", dest="calib_pred", default=False,
    #                  help='Use predictions on calibration items, rather than given labels: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
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
    parser.add_option('--run_all', action="store_true", dest="run_all", default=False,
                      help='Run models using combined train and calibration data: default=%default')
    parser.add_option('--verbose', action="store_true", dest="verbose", default=False,
                      help='Print more output: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    config_file = args[2]

    n_train = int(options.n_train)
    #n_calib = int(options.n_calib)
    sample_labels = options.sample
    suffix = options.suffix
    model_type = options.model
    #loss = options.loss
    loss = 'log'
    dh = int(options.dh)
    alpha_min = float(options.alpha_min)
    alpha_max = float(options.alpha_max)
    do_ensemble = True
    #exclude_calib = options.exclude_calib
    #calib_pred = options.calib_pred
    label = options.label
    penalty = options.penalty
    objective = options.objective
    intercept = not options.no_intercept
    n_dev_folds = int(options.n_dev_folds)
    repeats = int(options.repeats)
    seed = options.seed
    if options.seed is not None:
        seed = int(seed)
        np.random.seed(seed)
    run_all = options.run_all
    verbose = options.verbose

    average = 'micro'

    cross_train_and_eval(project_dir, subset, config_file, n_train, suffix, model_type, loss, do_ensemble, dh, label, penalty, intercept, n_dev_folds, repeats, verbose, average, objective, seed, alpha_min, alpha_max, sample_labels, run_all)


def cross_train_and_eval(project_dir, subset, config_file, n_train=500, suffix='', model_type='LR', loss='log', do_ensemble=True, dh=100, label='label', penalty='l1', intercept=True, n_dev_folds=5, repeats=1, verbose=False, average='micro', objective='f1', seed=None, alpha_min=0.01, alpha_max=1000.0, sample_labels=False, run_all=False):

    field_name = 'nosplit'
    model_basename = subset + '_' + label + '_' + field_name + '_' + model_type + '_' + penalty
    if model_type == 'MLP':
        model_basename += '_' + str(dh)
    model_basename += '_' + str(n_train) + '_' + objective
    model_basename += suffix

    # save the experiment parameters to a log file
    logfile = os.path.join(dirs.dir_logs(project_dir), model_basename + '.json')
    fh.makedirs(dirs.dir_logs(project_dir))
    log = {
        'project': project_dir,
        'subset': subset,
        'field_name': 'nosplit',
        'config_file': config_file,
        'n_train': n_train,
        'suffix': suffix,
        'model_type': model_type,
        'loss': loss,
        'dh': dh,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'do_ensemble': do_ensemble,
        'label': label,
        'penalty': penalty,
        'intercept': intercept,
        'objective': objective,
        'n_dev_folds': n_dev_folds,
        'repeats': repeats,
        'average': average,
        #'use_calib_pred': use_calib_pred,
        #'exclude_calib': exclude_calib,
        'sample_labels': sample_labels
    }
    fh.write_to_json(log, logfile)

    # load the features specified in the config file
    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    # load all labels
    label_dir = dirs.dir_labels(project_dir, subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
    n_items, n_classes = labels_df.shape

    weights_df = None

    # eliminate items with no labels
    print("Subsetting items with labels")
    label_sums_df = labels_df.sum(axis=1)
    labeled_item_selector = label_sums_df > 0
    labels_df = labels_df[labeled_item_selector]
    n_items, n_classes = labels_df.shape
    labeled_items = list(set(labels_df.index))

    print("Starting repeats")
    # repeat the following process multiple times with different random splits of train / calibration / test data
    for r in range(repeats):
        print("* Repetition %d *" % r)
        # take a random subset of the training data
        np.random.shuffle(labeled_items)
        train_items = labeled_items[:n_train]
        test_items = labeled_items[n_train:]
        n_test = len(test_items)
        n_calib = 0

        # create a data frame to hold a summary of the results
        output_df = pd.DataFrame([], columns=['N', 'training data', 'test data', 'cal', 'estimate', 'RMSE', '95lcl', '95ucl', 'contains_test'])
        # create a unique name ofr this model
        model_name = model_basename + '_' + 'nosplit' + '_' + str(r)

        print("Train: %d, calibration: %d, test: %d" % (n_train, n_calib, n_test))
        test_labels_df = labels_df.loc[test_items]

        # if instructed, sample labels in proportion to annotations (to simulate having one label per item)
        if sample_labels:
            print("Sampling labels")
            # normalize the labels
            temp = labels_df.values / np.array(labels_df.values.sum(axis=1).reshape((n_items, 1)), dtype=float)
            samples = np.zeros([n_items, n_classes], dtype=int)
            for i in range(n_items):
                index = np.random.choice(np.arange(n_classes), size=1, p=temp[i, :])
                samples[i, index] = 1
            sampled_labels_df = pd.DataFrame(samples, index=labels_df.index, columns=labels_df.columns)
        else:
            sampled_labels_df = labels_df

        train_labels_df = sampled_labels_df.loc[train_items].copy()

        # get the true proportion of labels in the test OR non-training data (calibration and test combined)
        target_props, target_estimate, target_std = get_estimate_and_std(labels_df)
        output_df.loc['target'] = [n_test, 'n/a', 'all', 'given', target_estimate, 0, target_estimate - 2 * target_std, target_estimate + 2 * target_std, np.nan]

        # get the same estimate from training data
        train_props, train_estimate, train_std = get_estimate_and_std(train_labels_df)
        # compute the error of this estimate
        train_rmse = np.sqrt((train_estimate - target_estimate)**2)
        train_contains_test = target_estimate > train_estimate - 2 * train_std and target_estimate < train_estimate + 2 * train_std
        output_df.loc['train'] = [n_train, 'train', 'train', 'n/a', train_estimate, train_rmse, np.nan, np.nan, np.nan]

        print("target proportions: (%0.3f, %0.3f); train proportions: %0.3f" % (target_estimate - 2 * target_std, target_estimate + 2 * target_std, train_estimate))

        if train_estimate > 0.5:
            pos_label = 0
        else:
            pos_label = 1
        print("Using %d as the positive label" % pos_label)

        results_df = pd.DataFrame([], columns=['f1', 'acc', 'calibration', 'calib overall'])

        # Now train a model on the training data, saving the calibration data for calibration
        print("Training model on training data only")
        model, dev_f1, dev_acc, dev_cal, dev_cal_overall = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max,  intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, pos_label=pos_label, verbose=verbose)
        results_df.loc['cross_val'] = [dev_f1, dev_acc, dev_cal, dev_cal_overall]

        # predict on test data
        test_predictions_df, test_pred_probs_df, test_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose)
        f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
        true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
        test_cal_rmse = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix())
        test_cal_rmse_overall = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
        results_df.loc['test'] = [f1_test, acc_test, test_cal_rmse, test_cal_rmse_overall]
        test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_pvc_estimate_internal = test_pred_proportions

        # predict on calibration and test data combined
        all_predictions_df, all_pred_probs_df, all_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=labeled_items, verbose=verbose)
        all_cc_estimate, all_pcc_estimate, all_acc_estimate_internal, all_pvc_estimate_internal = all_pred_proportions

        cc_rmse = np.sqrt((all_cc_estimate[1] - target_estimate)**2)
        pcc_rmse = np.sqrt((all_pcc_estimate[1] - target_estimate)**2)

        output_df.loc['CC_all'] = [n_items, 'train', 'all', 'predicted', all_cc_estimate[1], cc_rmse, np.nan, np.nan, np.nan]
        output_df.loc['PCC_all'] = [n_items, 'train', 'all', 'predicted', all_pcc_estimate[1], pcc_rmse, np.nan, np.nan, np.nan]

        averaged_cc_estimate = (test_cc_estimate[1] * n_test + train_estimate * n_train) / float(n_test + n_train)
        averaged_pcc_estimate = (test_pcc_estimate[1] * n_test + train_estimate * n_train) / float(n_test + n_train)

        averaged_cc_rmse = np.sqrt((averaged_cc_estimate - target_estimate)**2)
        averaged_pcc_rmse = np.sqrt((averaged_pcc_estimate - target_estimate)**2)

        output_df.loc['CC_nontrain_averaged'] = [n_items, 'train', 'all', 'given', averaged_cc_estimate, averaged_cc_rmse, np.nan, np.nan, np.nan]
        output_df.loc['PCC_nontrain_averaged'] = [n_items, 'train', 'all', 'given', averaged_pcc_estimate, averaged_pcc_rmse, np.nan, np.nan, np.nan]

        all_acc_rmse_internal = np.sqrt((all_acc_estimate_internal[1] - target_estimate) ** 2)
        all_pvc_rmse_internal = np.sqrt((all_pvc_estimate_internal[1] - target_estimate) ** 2)

        output_df.loc['ACC_internal'] = [n_items, 'train', 'all', 'predicted', all_acc_estimate_internal[1], all_acc_rmse_internal, np.nan, np.nan, np.nan]
        output_df.loc['PVC_internal'] = [n_items, 'train', 'all', 'predicted', all_pvc_estimate_internal[1], all_pvc_rmse_internal, np.nan, np.nan, np.nan]

        print("Venn internal all")
        all_pred_ranges_internal, all_preds_internal = ivap.estimate_probs_from_labels_internal(project_dir, model, model_name, subset, labeled_items)

        pred_range = np.mean(all_pred_ranges_internal, axis=0)
        venn_estimate = np.mean(all_preds_internal)

        venn_rmse = np.sqrt((venn_estimate - target_estimate)**2)
        venn_contains_test = pred_range[0] < target_estimate < pred_range[1]
        output_df.loc['Venn_internal'] = [n_items, 'train', 'all', 'predicted', venn_estimate, venn_rmse, pred_range[0], pred_range[1], venn_contains_test]

        print("Venn internal test")
        test_pred_ranges_internal, test_preds_internal = ivap.estimate_probs_from_labels_internal(project_dir, model, model_name, subset, test_items)

        pred_range = np.mean(test_pred_ranges_internal, axis=0)
        venn_estimate = (np.mean(test_preds_internal) * n_test + train_estimate * n_train) / float(n_test + n_train)
        venn_rmse = np.sqrt((venn_estimate - target_estimate)**2)

        averaged_lower = (pred_range[0] * n_test + (train_estimate - 2 * train_std) * n_calib) / float(n_test + n_train)
        averaged_upper = (pred_range[1] * n_test + (train_estimate + 2 * train_std) * n_calib) / float(n_test + n_train)
        venn_contains_test = averaged_lower < target_estimate < averaged_upper

        output_df.loc['Venn_internal_averaged'] = [n_items, 'train', 'all', 'given', venn_estimate, venn_rmse, averaged_lower, averaged_upper, venn_contains_test]

        results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))
        output_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))


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
