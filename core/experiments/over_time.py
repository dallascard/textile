import os
from optparse import OptionParser

import numpy as np
import pandas as pd

from core.util import file_handling as fh
from core.preprocessing import features
from core.main import train, predict, evaluate_predictions
from core.models import evaluation
from core.models import get_top_features
from core.discovery import fightin_words
from core.util import dirs


def main():
    usage = "%prog project_dir subset model_config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--stage2', dest='logfile', default=None,
                      help='run stage 2 using logfile from stage 1: default=%default')
    #parser.add_option('--n_train', dest='n_train', default=100,
    #                  help='Number of training instances to use (0 for all): default=%default')
    #parser.add_option('--n_calib', dest='n_calib', default=100,
    #                  help='Number of test instances to use for calibration: default=%default')
    parser.add_option('--first_year', dest='first_year', default=1996,
                      help='First year: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to mdoel name: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|MLP]: default=%default')
    parser.add_option('--DL', action="store_true", dest="DL", default=False,
                      help='Try a decision list: default=%default')
    parser.add_option('--loss', dest='loss', default='log',
                      help='Loss function [log|brier]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--nonlinearity',  dest='nonlinearity', default='tanh',
                      help='Nonlinearity for an MLP [tanh|sigmoid|relu]: default=%default')
    parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
                      help='Minimum value of training hyperparameter: default=%default')
    parser.add_option('--alpha_max', dest='alpha_max', default=1000,
                      help='Maximum value of training hyperparameter: default=%default')
    parser.add_option('--n_alphas', dest='n_alphas', default=8,
                      help='Number of alpha values to try: default=%default')
    #parser.add_option('--ensemble', action="store_true", dest="ensemble", default=False,
    #                  help='Make an ensemble from cross-validation, instead of training one model: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    #parser.add_option('--cshift', dest='cshift', default=None,
    #                  help='Covariate shift method [None|classify]: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--early', action="store_true", dest="early_stopping", default=False,
                      help='Use early stopping for MLP: default=%default')
    parser.add_option('--group', action="store_true", dest="group", default=False,
                      help='Group identical feature vectors: default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    #parser.add_option('--repeats', dest='repeats', default=3,
    #                  help='Number of repeats with random calibration/test splits: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')
    #parser.add_option('--run_all', action="store_true", dest="run_all", default=False,
    #                  help='Run models using combined train and calibration data: default=%default')
    parser.add_option('--n_terms', dest='n_terms', default=100,
                      help='Number of terms to select before intersection: default=%default')
    parser.add_option('--annotated', dest='annotated', default=None,
                      help='Annotated subset to load the corresponding features from just annotated text: default=%default')
    parser.add_option('--verbose', action="store_true", dest="verbose", default=False,
                      help='Print more output: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    config_file = args[2]

    stage1_logfile = options.logfile
    #n_train = int(options.n_train)
    #n_calib = int(options.n_calib)
    first_year = int(options.first_year)
    sample_labels = options.sample
    suffix = options.suffix
    model_type = options.model
    DL = options.DL
    loss = options.loss
    dh = int(options.dh)
    nonlinearity = options.nonlinearity
    alpha_min = float(options.alpha_min)
    alpha_max = float(options.alpha_max)
    n_alphas = int(options.n_alphas)
    do_ensemble = False
    #exclude_calib = options.exclude_calib
    #calib_pred = options.calib_pred
    label = options.label
    penalty = options.penalty
    #cshift = options.cshift
    objective = options.objective
    early_stopping = options.early_stopping
    intercept = not options.no_intercept
    group_identical = options.group
    n_dev_folds = int(options.n_dev_folds)
    #repeats = int(options.repeats)
    seed = options.seed
    if options.seed is not None:
        seed = int(seed)
        np.random.seed(seed)
    #run_all = options.run_all
    annotated = options.annotated
    n_terms = int(options.n_terms)
    verbose = options.verbose

    average = 'micro'

    test_over_time(project_dir, subset, config_file, first_year, stage1_logfile, penalty, suffix, model_type, loss, objective, do_ensemble, dh, label, intercept, n_dev_folds, verbose, average, seed, alpha_min, alpha_max, n_alphas, sample_labels, group_identical, annotated, n_terms, nonlinearity, early_stopping=early_stopping, DL=DL)

    #stage2(project_dir, subset, config_file, penalty, suffix, do_ensemble, dh, label, intercept, n_dev_folds, verbose, average, seed, alpha_min, alpha_max, sample_labels, annotated_subset=annotated, n_terms=n_terms)


def test_over_time(project_dir, subset, config_file, first_year, stage1_logfile=None, penalty='l2', suffix='', model_type='LR', loss='log', objective='f1', do_ensemble=True, dh=100, label='label', intercept=True, n_dev_folds=5, verbose=False, average='micro', seed=None, alpha_min=0.01, alpha_max=1000.0, n_alphas=8, sample_labels=False, group_identical=False, annotated_subset=None, n_terms=0, nonlinearity='tanh', init_lr=1e-4, min_epochs=2, max_epochs=100, patience=8, tol=1e-4, early_stopping=False, DL=False):
    # Just run a regular model, one per year, training on the past, and save the reults

    log = {
        'project': project_dir,
        'subset': subset,
        'config_file': config_file,
        'first_year': first_year,
        'stage1_logfile': stage1_logfile,
        'penalty': penalty,
        'suffix': suffix,
        'model_type': model_type,
        'loss': loss,
        'objective': objective,
        'do_ensemble': do_ensemble,
        'dh': dh,
        'label': label,
        'intercept': intercept,
        'n_dev_folds': n_dev_folds,
        'average': average,
        'seed': seed,
        'alpha_min': alpha_min,
        'alpha_max': alpha_max,
        'n_alphas': n_alphas,
        'sample_labels': sample_labels,
        'group_identical': group_identical,
        'annotated_subset': annotated_subset,
        'n_terms': n_terms,
        'nonlinearity': nonlinearity,
        'init_lr': init_lr,
        'min_epochs': min_epochs,
        'max_epochs': max_epochs,
        'patience': patience,
        'tol': tol,
        'early_stopping': early_stopping
    }

    model_basename = make_model_basename(log)
    stage1_model_basename = ''
    if stage1_logfile is not None:
        stage1_log = fh.read_json(stage1_logfile)
        stage1_model_basename = make_model_basename(stage1_log)

    # save the experiment parameters to a log file
    logfile = os.path.join(dirs.dir_logs(project_dir), model_basename + '.json')
    fh.makedirs(dirs.dir_logs(project_dir))

    fh.write_to_json(log, logfile)

    # load the features specified in the config file
    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    # load the file that contains metadata about each item
    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata['year'].values))
    field_vals.sort()
    print("Splitting data according to :", field_vals)

    # DEBUG:
    field_vals = ['2009']

    for target_year in field_vals:
        if int(target_year) >= first_year:
            print("\nTesting on %s" % target_year)
            model_name = model_basename + '_' + str(target_year)
            stage1_model_name = stage1_model_basename + '_' + str(target_year)
            # first, split into training and non-train data based on the field of interest

            ## DEBUG!
            test_selector_all = metadata['year'] >= int(target_year)
            test_subset_all = metadata[test_selector_all]
            test_items_all = test_subset_all.index.tolist()
            n_test_all = len(test_items_all)

            train_selector_all = metadata['year'] < int(target_year)
            train_subset_all = metadata[train_selector_all]
            train_items_all = list(train_subset_all.index)
            n_train_all = len(train_items_all)

            print("Test year: %d Train: %d, Test: %d (labeled and unlabeled)" % (int(target_year), n_train_all, n_test_all))

            # load all labels
            label_dir = dirs.dir_labels(project_dir, subset)
            labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
            n_items, n_classes = labels_df.shape

            vocab = None
            if stage1_logfile is not None:

                fightin_lexicon = None
                if annotated_subset is not None:
                    print("Determining fightin' words")
                    fightin_words.find_most_annotated_features(project_dir, annotated_subset, subset, config_file, items_to_use=train_items_all, remove_stopwords=False)
                    fightin_lexicon, scores = fightin_words.load_from_config_files(project_dir, annotated_subset, subset, config_file, items_to_use=train_items_all, n=n_terms, remove_stopwords=True)
                    fightin_lexicon_test, scores = fightin_words.load_from_config_files(project_dir, annotated_subset, subset, config_file, items_to_use=test_items_all, n=n_terms, remove_stopwords=True)
                    print(fightin_lexicon)
                    #print(fightin_lexicon_test)
                    #vocab = list(fightin_lexicon)
                    #vocab.sort()

                print("Loading feature from stage 1")
                # load features from previous model
                top_features = get_top_features.get_top_features(os.path.join(dirs.dir_models(project_dir), stage1_model_name), n_terms)
                lr_features, weights = zip(*top_features)
                vocab = list(lr_features)

                #if annotated_subset is not None:
                #    print("\nTaking intersection:")
                #    intersection = set(lr_features).intersection(set(fightin_lexicon))
                #    vocab = list(intersection)
                #    vocab.sort()
                #    for w in vocab:
                #        print(w)


                #vocab = [w for w in vocab if w not in stopwords]

                for w in vocab:
                    print(w)

                vocab.sort()

                #if annotated_subset is not None:
                #    print("Missing:")
                #    print(set(fightin_lexicon_test) - set(vocab))


            # add in a stage to eliminate items with no labels
            print("Subsetting items with labels")
            label_sums_df = labels_df.sum(axis=1)
            labeled_item_selector = label_sums_df > 0
            labels_df = labels_df[labeled_item_selector]
            n_items, n_classes = labels_df.shape
            labeled_items = set(labels_df.index)

            train_items = [i for i in train_items_all if i in labeled_items]
            test_items = [i for i in test_items_all if i in labeled_items]
            n_train = len(train_items)
            n_test = len(test_items)

            weights_df = None
            if weights_df is not None:
                weights_df = weights_df[labeled_item_selector]

            print("Labeled train: %d, test: %d" % (n_train, n_test))

            # create a data frame to hold a summary of the results
            output_df = pd.DataFrame([], columns=['N', 'training data', 'test data', 'cal', 'estimate', 'MAE', '95lcl', '95ucl', 'contains_test'])

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
            target_props, target_estimate, target_std = get_estimate_and_std(test_labels_df, use_n_annotations=True)

            output_df.loc['target'] = [n_test, 'test', 'test', 'n/a', target_estimate, 0, target_estimate - 2 * target_std, target_estimate + 2 * target_std, np.nan]

            # get the same estimate from training data
            train_props, train_estimate, train_std = get_estimate_and_std(train_labels_df, use_n_annotations=True)
            # compute the error of this estimate
            train_rmse = np.sqrt((train_estimate - target_estimate)**2)
            train_contains_test = target_estimate > train_estimate - 2 * train_std and target_estimate < train_estimate + 2 * train_std
            output_df.loc['train'] = [n_train, 'train', 'test', 'n/a', train_estimate, train_rmse, train_estimate - 2 * train_std, train_estimate + 2 * train_std, train_contains_test]

            #print("target proportions: (%0.3f, %0.3f); train proportions: %0.3f" % (target_estimate - 2 * target_std, target_estimate + 2 * target_std, train_estimate))

            if train_estimate > 0.5:
                pos_label = 0
            else:
                pos_label = 1
            print("Using %d as the positive label" % pos_label)

            results_df = pd.DataFrame([], columns=['f1', 'acc', 'mae', 'estimated calibration'])

            # Now train a model on the training data, saving the calibration data for calibration
            print("Training a model")
            model, dev_f1, dev_acc, dev_cal_mae, dev_cal_est = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items, penalty='l2', alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, pos_label=pos_label, vocab=vocab, group_identical=group_identical, nonlinearity=nonlinearity, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, tol=tol, early_stopping=early_stopping, verbose=verbose)
            results_df.loc['cross_val'] = [dev_f1, dev_acc, dev_cal_mae, dev_cal_est]

            # predict on test data
            force_dense = False
            if model_type == 'MLP':
                force_dense = True
            test_predictions_df, test_pred_probs_df, test_pred_proportions = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose, force_dense=force_dense, group_identical=group_identical)
            f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
            true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
            #test_cal_mae = evaluation.eval_proportions_mae(test_labels_df.as_matrix(), test_pred_probs_df.as_matrix())
            test_cal_est = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
            test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_pvc_estimate_internal = test_pred_proportions

            test_cc_mae = np.mean(np.abs(test_cc_estimate[1] - target_estimate))
            test_pcc_mae = np.mean(np.abs(test_pcc_estimate[1] - target_estimate))

            results_df.loc['test'] = [f1_test, acc_test, test_pcc_mae, test_cal_est]

            output_df.loc['CC_test'] = [n_train, 'train', 'test', 'n/a', test_cc_estimate[1], test_cc_mae, np.nan, np.nan, np.nan]
            output_df.loc['PCC_test'] = [n_train, 'train', 'test', 'n/a', test_pcc_estimate[1], test_pcc_mae, np.nan, np.nan, np.nan]

            test_acc_rmse_internal = np.sqrt((test_acc_estimate_internal[1] - target_estimate) ** 2)
            test_pvc_rmse_internal = np.sqrt((test_pvc_estimate_internal[1] - target_estimate) ** 2)

            output_df.loc['ACC_internal'] = [n_train, 'train', 'test', 'n/a', test_acc_estimate_internal[1], test_acc_rmse_internal, np.nan, np.nan, np.nan]
            output_df.loc['PVC_internal'] = [n_train, 'train', 'nontrain', 'predicted', test_pvc_estimate_internal[1], test_pvc_rmse_internal, np.nan, np.nan, np.nan]

            """
            if DL:
                print("Training a model")
                model_type = 'DL'
                DL_model_name = model_name + '_DL'
                model, _, _, _, _ = train.train_model_with_labels(project_dir, model_type, loss, DL_model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items, penalty='l2', alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, pos_label=pos_label, vocab=vocab, group_identical=group_identical, nonlinearity=nonlinearity, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, tol=tol, early_stopping=early_stopping, verbose=verbose)

                # predict on test data
                force_dense = False
                if model_type == 'MLP':
                    force_dense = True
                test_predictions_df, test_pred_probs_df, test_pred_proportions = predict.predict(project_dir, model, DL_model_name, subset, label, items_to_use=test_items, verbose=verbose, force_dense=force_dense, group_identical=group_identical)
                f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
                true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)

                #test_cal_mae = evaluation.eval_proportions_mae(test_labels_df.as_matrix(), test_pred_probs_df.as_matrix())
                test_cal_est = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
                test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_pvc_estimate_internal = test_pred_proportions

                test_cc_mae = np.mean(np.abs(test_cc_estimate[1] - target_estimate))
                test_pcc_mae = np.mean(np.abs(test_pcc_estimate[1] - target_estimate))

                output_df.loc['CC_test_DL'] = [n_train, 'train', 'test', 'n/a', test_cc_estimate[1], test_cc_mae, np.nan, np.nan, np.nan]
                output_df.loc['PCC_test_DL'] = [n_train, 'train', 'test', 'n/a', test_pcc_estimate[1], test_pcc_mae, np.nan, np.nan, np.nan]
            """

            results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))
            output_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))




def make_model_basename(log):
    model_basename = log['subset'] + '_' + log['label'] + '_' + 'year' + '_' + log['model_type'] + '_' + log['penalty']
    model_basename += '_' + str(log['dh']) + '_' + log['objective']
    if log['sample_labels']:
        model_basename += '_sampled'
    model_basename += log['suffix']
    if log['group_identical']:
        model_basename += '_grouped'
    if log['annotated_subset']:
        model_basename += '_fight'
    if log['stage1_logfile'] is not None:
        model_basename += '_stage2_' + str(log['n_terms'])
    return model_basename


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
