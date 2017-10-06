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
    usage = "%prog project_dir subset cross_field_name config.json reference_model_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default=0,
                      help='Number of training instances to use (0 for all): default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=0,
                      help='Number of test instances to use for calibration: default=%default')
    #parser.add_option('--sample', action="store_true", dest="sample", default=False,
    #                  help='Sample labels instead of averaging: default=%default')
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to mdoel name: default=%default')
    parser.add_option('--model', dest='model', default='MLP',
                      help='Model type [SGD|MLP]: default=%default')
    parser.add_option('--dh', dest='dh', default=0,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    #parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
    #                  help='Minimum value of training hyperparameter: default=%default')
    #parser.add_option('--alpha_max', dest='alpha_max', default=1000,
    #                  help='Maximum value of training hyperparameter: default=%default')
    #parser.add_option('--ensemble', action="store_true", dest="ensemble", default=False,
    #                  help='Make an ensemble from cross-validation, instead of training one model: default=%default')
    #parser.add_option('--exclude_calib', action="store_true", dest="exclude_calib", default=False,
    #                  help='Exclude the calibration data from the evalaution: default=%default')
    #parser.add_option('--calib_pred', action="store_true", dest="calib_pred", default=False,
    #                  help='Use predictions on calibration items, rather than given labels: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    #parser.add_option('--cshift', dest='cshift', default=None,
    #                  help='Covariate shift method [None|classify]: default=%default')
    #parser.add_option('--penalty', dest='penalty', default='l2',
    #                  help='Regularization type: default=%default')
    #parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
    #                  help='Use to fit a model with no intercept: default=%default')
    #parser.add_option('--objective', dest='objective', default='f1',
    #                  help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--repeats', dest='repeats', default=3,
                      help='Number of repeats with random calibration/test splits: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')
    #parser.add_option('--run_all', action="store_true", dest="run_all", default=False,
    #                  help='Run models using combined train and calibration data: default=%default')
    parser.add_option('--verbose', action="store_true", dest="verbose", default=False,
                      help='Print more output: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    field_name = args[2]
    config_file = args[3]
    reference_model_dir = args[4]

    n_train = int(options.n_train)
    n_calib = int(options.n_calib)
    #sample_labels = options.sample
    suffix = options.suffix
    model_type = options.model
    loss = 'brier'
    dh = int(options.dh)
    #alpha_min = float(options.alpha_min)
    #alpha_max = float(options.alpha_max)
    do_ensemble = True
    #exclude_calib = options.exclude_calib
    #calib_pred = options.calib_pred
    label = options.label
    #penalty = options.penalty
    #cshift = options.cshift
    #objective = options.objective
    #intercept = not options.no_intercept
    n_dev_folds = int(options.n_dev_folds)
    repeats = int(options.repeats)
    seed = options.seed
    if options.seed is not None:
        seed = int(seed)
        np.random.seed(seed)
    #run_all = options.run_all
    verbose = options.verbose

    average = 'micro'

    cross_train_and_eval(project_dir, reference_model_dir, subset, field_name, config_file, n_calib, n_train, suffix, model_type, loss, do_ensemble, dh, label, n_dev_folds, repeats, verbose, average, seed, )


def cross_train_and_eval(project_dir, reference_model_dir, subset, field_name, config_file, n_calib=0, n_train=100, suffix='', model_type='LR', loss='log', do_ensemble=True, dh=100, label='label', n_dev_folds=5, repeats=1, verbose=False, average='micro', seed=None):

    model_basename = subset + '_' + label + '_' + field_name + '_' + model_type
    if model_type == 'MLP':
        model_basename += '_' + str(dh)
    model_basename +=  '_' + str(n_train) + '_' + str(n_calib)
    model_basename += suffix

    # save the experiment parameters to a log file
    logfile = os.path.join(dirs.dir_logs(project_dir), model_basename + '.json')
    fh.makedirs(dirs.dir_logs(project_dir))
    log = {
        'project': project_dir,
        'subset': subset,
        'field_name': field_name,
        'config_file': config_file,
        'n_calib': n_calib,
        'n_train': n_train,
        'suffix': suffix,
        'model_type': model_type,
        'loss': loss,
        'dh': dh,
        'do_ensemble': do_ensemble,
        'label': label,
        'n_dev_folds': n_dev_folds,
        'repeats': repeats,
        'average': average
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
    print("Splitting data according to :", field_vals)

    # repeat the following value for each fold of the partition of interest (up to max_folds, if given)
    for v_i, v in enumerate(field_vals):
        print("\nTesting on %s" % v)
        # first, split into training and non-train data based on the field of interest
        train_selector = metadata[field_name] != v
        train_subset = metadata[train_selector]
        train_items = list(train_subset.index)
        n_train_cshift = len(train_items)

        non_train_selector = metadata[field_name] == v
        non_train_subset = metadata[non_train_selector]
        non_train_items = non_train_subset.index.tolist()
        n_non_train_cshift = len(non_train_items)

        print("Train: %d, non-train: %d" % (n_train_cshift, n_non_train_cshift))

        # load all labels
        label_dir = dirs.dir_labels(project_dir, subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
        n_items, n_classes = labels_df.shape

        weights_df = None

        # add in a stage to eliminate items with no labels?
        print("Subsetting items with labels")
        label_sums_df = labels_df.sum(axis=1)
        labeled_item_selector = label_sums_df > 0
        labels_df = labels_df[labeled_item_selector]
        n_items, n_classes = labels_df.shape
        labeled_items = set(labels_df.index)

        train_items = [i for i in train_items if i in labeled_items]
        non_train_items = [i for i in non_train_items if i in labeled_items]
        n_non_train = len(non_train_items)

        if weights_df is not None:
            weights_df = weights_df[labeled_item_selector]

        print("Starting repeats")
        # repeat the following process multiple times with different random splits of train / calibration / test data
        for r in range(repeats):
            print("* Repetition %d *" % r)
            # next, take a random subset of the training data (and ignore the rest), to simulate fewer annotated items
            if n_train > 0:
                np.random.shuffle(train_items)
                train_items_r = np.random.choice(train_items, size=n_train, replace=False)
            else:
                train_items_r = train_items

            n_train_r = len(train_items_r)

            # create a data frame to hold a summary of the results
            output_df = pd.DataFrame([], columns=['N', 'training data', 'test data', 'cal', 'estimate', 'RMSE', '95lcl', '95ucl', 'contains_test'])
            # create a unique name ofr this model
            model_name = model_basename + '_' + str(v) + '_' + str(r)

            # now, divide the non-train data into a calibration and a test set
            #n_calib = int(calib_prop * n_non_train)
            np.random.shuffle(non_train_items)
            if n_calib > n_non_train:
                n_calib = int(n_non_train / 2)
                print("Warning!!: only %d non-train items; using 1/2 for calibration" % n_non_train)

            calib_items = non_train_items[:n_calib]
            test_items = non_train_items[n_calib:]
            n_test = len(test_items)

            print("Train: %d, calibration: %d, test: %d" % (n_train_r, n_calib, n_test))
            test_labels_df = labels_df.loc[test_items]
            non_train_labels_df = labels_df.loc[non_train_items]

            sampled_labels_df = labels_df

            train_labels_r_df = sampled_labels_df.loc[train_items_r].copy()
            calib_labels_df = sampled_labels_df.loc[calib_items].copy()

            # get the true proportion of labels in the test OR non-training data (calibration and test combined)
            target_props, target_estimate, target_std = get_estimate_and_std(non_train_labels_df)
            output_df.loc['target'] = [n_test, 'nontrain', 'nontrain', 'given', target_estimate, 0, target_estimate - 2 * target_std, target_estimate + 2 * target_std, np.nan]

            # get the same estimate from training data
            train_props, train_estimate, train_std = get_estimate_and_std(train_labels_r_df)
            # compute the error of this estimate
            train_rmse = np.sqrt((train_estimate - target_estimate)**2)
            train_contains_test = target_estimate > train_estimate - 2 * train_std and target_estimate < train_estimate + 2 * train_std
            output_df.loc['train'] = [n_train_r, 'train', 'train', 'n/a', train_estimate, train_rmse, np.nan, np.nan, np.nan]

            print("target proportions: (%0.3f, %0.3f); train proportions: %0.3f" % (target_estimate - 2 * target_std, target_estimate + 2 * target_std, train_estimate))

            if train_estimate > 0.5:
                pos_label = 0
            else:
                pos_label = 1
            print("Using %d as the positive label" % pos_label)

            # repeat for labeled calibration data
            if n_calib > 0:
                calib_props, calib_estimate, calib_std = get_estimate_and_std(calib_labels_df)
                calib_rmse = np.sqrt((calib_estimate - target_estimate)**2)
                # check if the test estimate is within 2 standard deviations of the estimate
                calib_contains_test = target_estimate > calib_estimate - 2 * calib_std and calib_estimate < calib_estimate + 2 * calib_std
                output_df.loc['calibration'] = [n_calib, 'calibration', 'nontrain', 'given', calib_estimate, calib_rmse, calib_estimate - 2 * calib_std, calib_estimate + 2 * calib_std, calib_contains_test]

                # do a test using the number of annotations rather than the number of items
                calib_props2, calib_estimate2, calib_std2 = get_estimate_and_std(calib_labels_df, use_n_annotations=True)
                calib_rmse2 = np.sqrt((calib_estimate2 - target_estimate)**2)
                calib_contains_test2 = target_estimate > calib_estimate2 - 2 * calib_std2 and calib_estimate < calib_estimate2 + 2 * calib_std2
                output_df.loc['calibration_n_annotations'] = [n_calib, 'calibration', 'nontrain', 'given', calib_estimate2, calib_rmse2, calib_estimate2 - 2 * calib_std2, calib_estimate2 + 2 * calib_std2, calib_contains_test2]

            results_df = pd.DataFrame([], columns=['f1', 'acc', 'calibration', 'calib overall'])

            # Now train a model on the training data, saving the calibration data for calibration

            print("Training model on training data only")
            model, dev_f1, dev_acc, dev_cal, dev_cal_overall = train.train_mlp_restricted(project_dir, reference_model_dir, model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items_r, intercept=True, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, pos_label=pos_label, verbose=verbose)
            results_df.loc['cross_val'] = [dev_f1, dev_acc, dev_cal, dev_cal_overall]

            # predict on calibration data
            if n_calib > 0:
                calib_predictions_df, calib_pred_probs_df, calib_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=calib_items, verbose=verbose, force_dense=True)
                calib_cc, calib_pcc, calib_acc, calib_pvc = calib_pred_proportions
                f1_cal, acc_cal = evaluate_predictions.evaluate_predictions(calib_labels_df, calib_predictions_df, calib_pred_probs_df, pos_label=pos_label, average=average, verbose=False)
                true_calib_vector = np.argmax(calib_labels_df.as_matrix(), axis=1)
                calib_cal_rmse = evaluation.evaluate_calibration_rmse(true_calib_vector, calib_pred_probs_df.as_matrix())
                calib_cal_rmse_overall = evaluation.evaluate_calibration_rmse(true_calib_vector, calib_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
                results_df.loc['calibration'] = [f1_cal, acc_cal, calib_cal_rmse, calib_cal_rmse_overall]

            # predict on test data
            test_predictions_df, test_pred_probs_df, test_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose, force_dense=True)
            f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
            true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
            test_cal_rmse = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix())
            test_cal_rmse_overall = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
            results_df.loc['test'] = [f1_test, acc_test, test_cal_rmse, test_cal_rmse_overall]
            test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_pvc_estimate_internal = test_pred_proportions


            # predict on calibration and test data combined
            nontrain_predictions_df, nontrain_pred_probs_df, nontrain_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=non_train_items, verbose=verbose, force_dense=True)
            nontrain_cc_estimate, nontrain_pcc_estimate, nontrain_acc_estimate_internal, nontrain_pvc_estimate_internal = nontrain_pred_proportions

            if n_calib > 0:
                cc_calib_rmse = np.sqrt((calib_cc[1] - calib_estimate)**2)
                output_df.loc['CC_cal'] = [n_non_train, 'train', 'calibration', 'predicted', calib_cc[1], cc_calib_rmse, np.nan, np.nan, np.nan]

                pcc_calib_rmse = np.sqrt((calib_pcc[1] - calib_estimate)**2)
                output_df.loc['PCC_cal'] = [n_non_train, 'train', 'calibration', 'predicted', calib_pcc[1], pcc_calib_rmse, np.nan, np.nan, np.nan]

            cc_rmse = np.sqrt((nontrain_cc_estimate[1] - target_estimate)**2)
            pcc_rmse = np.sqrt((nontrain_pcc_estimate[1] - target_estimate)**2)

            output_df.loc['CC_nontrain'] = [n_non_train, 'train', 'nontrain', 'predicted', nontrain_cc_estimate[1], cc_rmse, np.nan, np.nan, np.nan]
            output_df.loc['PCC_nontrain'] = [n_non_train, 'train', 'nontrain', 'predicted', nontrain_pcc_estimate[1], pcc_rmse, np.nan, np.nan, np.nan]

            if n_calib > 0:
                averaged_cc_estimate = (test_cc_estimate[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                averaged_pcc_estimate = (test_pcc_estimate[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)

                averaged_cc_rmse = np.sqrt((averaged_cc_estimate - target_estimate)**2)
                averaged_pcc_rmse = np.sqrt((averaged_pcc_estimate - target_estimate)**2)

                output_df.loc['CC_nontrain_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_cc_estimate, averaged_cc_rmse, np.nan, np.nan, np.nan]
                output_df.loc['PCC_nontrain_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_pcc_estimate, averaged_pcc_rmse, np.nan, np.nan, np.nan]

            """
            nontrain_acc_rmse_internal = np.sqrt((nontrain_acc_estimate_internal[1] - target_estimate) ** 2)
            nontrain_pvc_rmse_internal = np.sqrt((nontrain_pvc_estimate_internal[1] - target_estimate) ** 2)

            output_df.loc['ACC_internal'] = [n_non_train, 'train', 'nontrain', 'predicted', nontrain_acc_estimate_internal[1], nontrain_acc_rmse_internal, np.nan, np.nan, np.nan]
            output_df.loc['PVC_internal'] = [n_non_train, 'train', 'nontrain', 'predicted', nontrain_pvc_estimate_internal[1], nontrain_pvc_rmse_internal, np.nan, np.nan, np.nan]

            if n_calib > 0:
                averaged_acc_estimate_internal = (test_acc_estimate_internal[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                averaged_pvc_estimate_internal = (test_pvc_estimate_internal[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                averaged_acc_rmse_internal = np.sqrt((averaged_acc_estimate_internal - target_estimate) ** 2)
                averaged_pvc_rmse_internal = np.sqrt((averaged_pvc_estimate_internal - target_estimate) ** 2)

                output_df.loc['ACC_internal_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_acc_estimate_internal, averaged_acc_rmse_internal, np.nan, np.nan, np.nan]
                output_df.loc['PVC_internal_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_pvc_estimate_internal, averaged_pvc_rmse_internal, np.nan, np.nan, np.nan]

            # do calibration here using calibration data
            if n_calib > 0:
                # expand the data so as to only have singly-labeled, weighted items
                _, calib_labels, calib_weights, calib_predictions = train.prepare_data(np.zeros([n_calib, 2]), calib_labels_df.values, predictions=calib_predictions_df.values)

                #calib_labels_expanded, calib_weights_expanded, calib_predictions_expanded = expand_labels(calib_labels.values, calib_predictions.values)
                acc = calibration.compute_acc(calib_labels, calib_predictions, n_classes, weights=calib_weights)
                acc_corrected = calibration.apply_acc_binary(nontrain_predictions_df.values, acc)
                acc_estimate = acc_corrected[1]
                acc_rmse = np.sqrt((acc_estimate - target_estimate) ** 2)
                output_df.loc['ACC'] = [n_non_train, 'train', 'nontrain', 'predicted', acc_estimate, acc_rmse, np.nan, np.nan, np.nan]

                pvc = calibration.compute_pvc(calib_labels, calib_predictions, n_classes, weights=calib_weights)
                pvc_corrected = calibration.apply_pvc(nontrain_predictions_df.values, pvc)
                pvc_estimate = pvc_corrected[1]
                pvc_rmse = np.sqrt((pvc_estimate - target_estimate) ** 2)
                output_df.loc['PVC'] = [n_non_train, 'train', 'nontrain', 'predicted', pvc_estimate, pvc_rmse, np.nan, np.nan, np.nan]

                acc_corrected = calibration.apply_acc_binary(test_predictions_df.values, acc)
                acc_estimate = acc_corrected[1]
                averaged_acc_estimate = (acc_estimate * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                averaged_acc_rmse = np.sqrt((acc_estimate - target_estimate) ** 2)
                output_df.loc['ACC_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_acc_estimate, averaged_acc_rmse, np.nan, np.nan, np.nan]

                pvc_corrected = calibration.apply_pvc(test_predictions_df.values, pvc)
                pvc_estimate = pvc_corrected[1]
                averaged_pvc_estimate = (pvc_estimate * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                averaged_pvc_rmse = np.sqrt((pvc_estimate - target_estimate) ** 2)
                output_df.loc['PVC_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_pvc_estimate, averaged_pvc_rmse, np.nan, np.nan, np.nan]

            print("Venn internal nontrain")
            #models = list(model._models.values())
            nontrain_pred_ranges_internal, nontrain_preds_internal = ivap.estimate_probs_from_labels_internal(project_dir, model, model_name, subset, non_train_items)

            pred_range = np.mean(nontrain_pred_ranges_internal, axis=0)
            venn_estimate = np.mean(nontrain_preds_internal)

            venn_rmse = np.sqrt((venn_estimate - target_estimate)**2)
            venn_contains_test = pred_range[0] < target_estimate < pred_range[1]
            output_df.loc['Venn_internal'] = [n_non_train, 'train', 'nontrain', 'predicted', venn_estimate, venn_rmse, pred_range[0], pred_range[1], venn_contains_test]

            if n_calib > 0:
                print("Venn internal test")
                test_pred_ranges_internal, test_preds_internal = ivap.estimate_probs_from_labels_internal(project_dir, model, model_name, subset, test_items)

                pred_range = np.mean(test_pred_ranges_internal, axis=0)
                venn_estimate = (np.mean(test_preds_internal) * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                venn_rmse = np.sqrt((venn_estimate - target_estimate)**2)

                averaged_lower = (pred_range[0] * n_test + (calib_estimate - 2 * calib_std) * n_calib) / float(n_test + n_calib)
                averaged_upper = (pred_range[1] * n_test + (calib_estimate + 2 * calib_std) * n_calib) / float(n_test + n_calib)
                venn_contains_test = averaged_lower < target_estimate < averaged_upper

                output_df.loc['Venn_internal_averaged'] = [n_non_train, 'train', 'nontrain', 'given', venn_estimate, venn_rmse, averaged_lower, averaged_upper, venn_contains_test]

                # Venn prediction using proper calibration data
                print("Venn calibration")
                calib_pred_ranges, calib_preds, calib_props_in_range, list_of_n_levels = ivap.estimate_probs_from_labels_cv(project_dir, model, model_name, sampled_labels_df, subset, calib_items=calib_items)
                print("Venn test")
                test_pred_ranges, test_preds = ivap.estimate_probs_from_labels(project_dir, model, model_name, sampled_labels_df, subset, subset, calib_items=calib_items, test_items=test_items)

                nontrain_pred_ranges = np.vstack([calib_pred_ranges, test_pred_ranges])
                nontrain_preds = np.r_[calib_preds, test_preds]

                nontrain_pred_range = np.mean(nontrain_pred_ranges, axis=0)
                nontrain_venn_estimate = np.mean(nontrain_preds)
                nontrain_venn_rmse = np.sqrt((nontrain_venn_estimate - target_estimate)**2)
                nontrain_contains_test = nontrain_pred_range[0] < target_estimate < nontrain_pred_range[1]
                output_df.loc['Venn'] = [n_non_train, 'train', 'nontrain', 'predicted', nontrain_venn_estimate, nontrain_venn_rmse, nontrain_pred_range[0], nontrain_pred_range[1], nontrain_contains_test]

                test_pred_range = np.mean(test_pred_ranges, axis=0)
                averaged_venn_estimate = (np.mean(test_preds) * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                averaged_venn_rmse = np.sqrt((averaged_venn_estimate - target_estimate)**2)

                averaged_lower = (test_pred_range[0] * n_test + (calib_estimate - 2 * calib_std) * n_calib) / float(n_test + n_calib)
                averaged_upper = (test_pred_range[1] * n_test + (calib_estimate + 2 * calib_std) * n_calib) / float(n_test + n_calib)
                venn_contains_test = averaged_lower < target_estimate < averaged_upper

                output_df.loc['Venn_averaged'] = [n_non_train, 'train', 'nontrain', 'given', averaged_venn_estimate, averaged_venn_rmse, averaged_lower, averaged_upper, venn_contains_test]

                fh.write_list_to_text(calib_props_in_range, os.path.join(dirs.dir_models(project_dir), model_name, 'venn_calib_props_in_range.csv'))
                fh.write_list_to_text(list_of_n_levels, os.path.join(dirs.dir_models(project_dir), model_name, 'list_of_n_levels.csv'))
                results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))

            # now train a model on the training and calibration data combined
            if run_all:
                print("Training model on all labeled data")
                calib_and_train_items_r = np.array(list(calib_items) + list(train_items_r))
                model, dev_f1, dev_acc, dev_cal, dev_cal_overall = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=calib_and_train_items_r, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, pos_label=pos_label, verbose=verbose)
                results_df.loc['cross_val_all'] = [dev_f1, dev_acc, dev_cal, dev_cal_overall]

                # get labels for test data
                test_predictions_df, test_pred_probs_df, test_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose)
                f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
                test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_pvc_estimate_internal = test_pred_proportions
                true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
                test_cal_rmse = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix())
                results_df.loc['test'] = [f1_test, acc_test, test_cal_rmse, 0]
                results_df.loc['test_all'] = [f1_test, acc_test, test_cal_rmse, 0]

                nontrain_predictions_df, nontrain_pred_probs_df, nontrain_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=non_train_items, verbose=verbose)
                nontrain_cc_estimate, nontrain_pcc_estimate, nontrain_acc_estimate_internal, nontrain_pvc_estimate_internal = nontrain_pred_proportions

                cc_rmse = np.sqrt((nontrain_cc_estimate[1] - target_estimate)**2)
                pcc_rmse = np.sqrt((nontrain_pcc_estimate[1] - target_estimate)**2)

                output_df.loc['CC_nontrain_all'] = [n_non_train, 'nontest', 'nontrain', 'predicted', nontrain_cc_estimate[1], cc_rmse, np.nan, np.nan, np.nan]
                output_df.loc['PCC_nontrain_all'] = [n_non_train, 'nontest', 'nontrain', 'predicted', nontrain_pcc_estimate[1], pcc_rmse, np.nan, np.nan, np.nan]

                if n_calib > 0:
                    averaged_cc_estimate = (test_cc_estimate[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                    averaged_pcc_estimate = (test_pcc_estimate[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)

                    averaged_cc_rmse = np.sqrt((averaged_cc_estimate - target_estimate)**2)
                    averaged_pcc_rmse = np.sqrt((averaged_pcc_estimate - target_estimate)**2)

                    output_df.loc['CC_nontrain_averaged_all'] = [n_non_train, 'nontest', 'nontrain', 'given', averaged_cc_estimate, averaged_cc_rmse, np.nan, np.nan, np.nan]
                    output_df.loc['PCC_nontrain_averaged_all'] = [n_non_train, 'nontest', 'nontrain', 'given', averaged_pcc_estimate, averaged_pcc_rmse, np.nan, np.nan, np.nan]

                nontrain_acc_rmse_internal = np.sqrt((nontrain_acc_estimate_internal[1] - target_estimate) ** 2)
                nontrain_pvc_rmse_internal = np.sqrt((nontrain_pvc_estimate_internal[1] - target_estimate) ** 2)

                output_df.loc['ACC_internal_all'] = [n_non_train, 'nontest', 'nontrain', 'predicted', nontrain_acc_estimate_internal[1], nontrain_acc_rmse_internal, np.nan, np.nan, np.nan]
                output_df.loc['PVC_internal_all'] = [n_non_train, 'nontest', 'nontrain', 'predicted', nontrain_pvc_estimate_internal[1], nontrain_pvc_rmse_internal, np.nan, np.nan, np.nan]

                if n_calib > 0:
                    averaged_acc_estimate_internal = (test_acc_estimate_internal[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                    averaged_pvc_estimate_internal = (test_pvc_estimate_internal[1] * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                    averaged_acc_rmse_internal = np.sqrt((averaged_acc_estimate_internal - target_estimate) ** 2)
                    averaged_pvc_rmse_internal = np.sqrt((averaged_pvc_estimate_internal - target_estimate) ** 2)

                    output_df.loc['ACC_internal_averaged_all'] = [n_non_train, 'nontest', 'nontrain', 'given', averaged_acc_estimate_internal, averaged_acc_rmse_internal, np.nan, np.nan, np.nan]
                    output_df.loc['PVC_internal_averaged_all'] = [n_non_train, 'nontest', 'nontrain', 'given', averaged_pvc_estimate_internal, averaged_pvc_rmse_internal, np.nan, np.nan, np.nan]

                print("Venn internal nontrain")
                nontrain_pred_ranges_internal, nontrain_preds_internal = ivap.estimate_probs_from_labels_internal(project_dir, model, model_name, subset, non_train_items)

                pred_range = np.mean(nontrain_pred_ranges_internal, axis=0)
                venn_estimate = np.mean(nontrain_preds_internal)

                venn_rmse = np.sqrt((venn_estimate - target_estimate)**2)
                venn_contains_test = pred_range[0] < target_estimate < pred_range[1]
                output_df.loc['Venn_internal_all'] = [n_non_train, 'nontest', 'nontrain', 'predicted', venn_estimate, venn_rmse, pred_range[0], pred_range[1], venn_contains_test]

                if n_calib > 0:
                    print("Venn internal test")
                    test_pred_ranges_internal, test_preds_internal = ivap.estimate_probs_from_labels_internal(project_dir, model, model_name, subset, test_items)

                    pred_range = np.mean(test_pred_ranges_internal, axis=0)
                    venn_estimate = (np.mean(test_preds_internal) * n_test + calib_estimate * n_calib) / float(n_test + n_calib)
                    venn_rmse = np.sqrt((venn_estimate - target_estimate)**2)

                    averaged_lower = (pred_range[0] * n_test + (calib_estimate - 2 * calib_std) * n_calib) / float(n_test + n_calib)
                    averaged_upper = (pred_range[1] * n_test + (calib_estimate + 2 * calib_std) * n_calib) / float(n_test + n_calib)
                    venn_contains_test = averaged_lower < target_estimate < averaged_upper

                    output_df.loc['Venn_internal_averaged_all'] = [n_non_train, 'nontest', 'nontrain', 'given', venn_estimate, venn_rmse, averaged_lower, averaged_upper, venn_contains_test]

            """
            results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))
            output_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))


def get_estimate_and_std(labels_df, use_n_annotations=False):
    n_items, n_classes = labels_df.shape
    assert n_classes == 2
    labels = labels_df.values.copy()

    if use_n_annotations:
        # treat each annotation as a separate sample
        n = np.sum(labels)
        # take the mean of all annotations
        props = np.sum(labels, axis=0) / float(n)
    else:
        # treat each document as a separate sample with a single label
        n = n_items
        # normalize the labels across classes
        labels = labels / np.reshape(labels.sum(axis=1), (len(labels), 1))
        # take the mean across items
        props = np.mean(labels, axis=0)

    # get the estimated probability of a positive label
    estimate = props[1]
    # estimate the variance by pretending this is a binomial distribution
    std = np.sqrt(estimate * (1 - estimate) / float(n))
    return props, estimate, std


if __name__ == '__main__':
    main()
