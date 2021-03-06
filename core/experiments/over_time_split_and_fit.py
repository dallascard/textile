import os
from optparse import OptionParser

import numpy as np
import pandas as pd

from core.util import file_handling as fh
from core.preprocessing import features
from core.main import train, predict, evaluate_predictions
from core.models import evaluation
from core.util import dirs


def main():
    usage = "%prog project_dir subset model_config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default=None,
                      help='Number of training instances to use (0 for all): default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=0,
                      help='Number of test instances to use for calibration: default=%default')
    parser.add_option('--field', dest='field', default='year',
                      help='Field on which to split for train and test: default=%default')
    parser.add_option('--train_start', dest='train_start', default=None,
                      help='Start of training range (all before test if None): default=%default')
    parser.add_option('--train_end', dest='train_end', default=None,
                      help='End of training range (all before test if None): default=%default')
    parser.add_option('--test_start', dest='test_start', default=2011,
                      help='Use training data from before this field value: default=%default')
    parser.add_option('--test_end', dest='test_end', default=2012,
                      help='Last field value of test data to use: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to mdoel name: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|DAN]: default=%default')
    parser.add_option('--loss', dest='loss', default='log',
                      help='Loss function [log|brier]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    parser.add_option('--dh', dest='dh', default=300,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--nonlinearity', dest='nonlinearity', default='tanh',
                      help='Nonlinearity for an MLP [tanh|sigmoid|relu]: default=%default')
    parser.add_option('--ls', dest='ls', default=10,
                      help='List size (for DL): default=%default')
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
    parser.add_option('--cshift', action="store_true", dest="cshift", default=False,
                      help='Covariate shift method [None|classify]: default=%default')
    parser.add_option('--n_cshift', dest='n_cshift', default=None,
                      help='Number of data points to use for covariate shift model: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    #parser.add_option('--early', action="store_true", dest="early_stopping", default=False,
    #                  help='Use early stopping for MLP: default=%default')
    parser.add_option('--group', action="store_true", dest="group", default=False,
                      help='Group identical feature vectors: default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--repeats', dest='repeats', default=1,
                      help='Number of repeats with random calibration/test splits: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')
    parser.add_option('--annotated', dest='annotated', default=None,
                      help='Annotated subset to load the corresponding features from just annotated text: default=%default')
    parser.add_option('--oracle', action="store_true", dest="oracle", default=False,
                      help='Try oracle selection for decision list: default=%default')
    parser.add_option('--lower', dest='lower', default=None,
                      help='Lower bound on LR weights: default=%default')
    parser.add_option('--interactive', action="store_true", dest="interactive", default=False,
                      help='Do interactive feature selection: default=%default')
    parser.add_option('--stoplist', dest='stoplist_file', default=None,
                      help='Stoplist file: default=%default')
    parser.add_option('--dropout', dest='dropout', default=0.0,
                      help='Apply word dropout to DANs: default=%default')
    parser.add_option('--lr', dest='init_lr', default=0.01,
                      help='Initial learning rate for DAN training: default=%default')
    parser.add_option('--patience', dest='patience', default=5,
                      help='Patience for DAN training: default=%default')
    parser.add_option('--max_epochs', dest='max_epochs', default=50,
                      help='Maximum number of epochs for DAN training: default=%default')
    parser.add_option('--test_prop', dest='test_prop', default=None,
                      help='Create a dataset with artificial test proportions: default=%default')
    parser.add_option('--verbose', action="store_true", dest="verbose", default=False,
                      help='Print more output: default=%default')


    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    config_file = args[2]

    n_train = options.n_train
    if n_train is not None:
        n_train = int(n_train)
    n_calib = int(options.n_calib)
    field = options.field
    train_start = options.train_start
    if train_start is not None:
        train_start = int(train_start)
    train_end = options.train_end
    if train_end is not None:
        train_end = int(train_end)
    test_start = int(options.test_start)
    test_end = int(options.test_end)
    sample_labels = options.sample
    suffix = options.suffix
    model_type = options.model
    loss = options.loss
    dh = int(options.dh)
    ls = int(options.ls)
    nonlinearity = options.nonlinearity
    alpha_min = float(options.alpha_min)
    alpha_max = float(options.alpha_max)
    n_alphas = int(options.n_alphas)

    do_ensemble = True

    #exclude_calib = options.exclude_calib
    #calib_pred = options.calib_pred
    label = options.label
    penalty = options.penalty
    cshift = options.cshift
    n_cshift = options.n_cshift
    if n_cshift is not None:
        n_cshift = int(n_cshift)
    objective = options.objective
    intercept = not options.no_intercept
    group_identical = options.group
    n_dev_folds = int(options.n_dev_folds)
    repeats = int(options.repeats)
    seed = options.seed
    #run_all = options.run_all
    annotated = options.annotated
    oracle = options.oracle
    lower = options.lower
    if lower is not None:
        lower = float(lower)
    interactive = options.interactive
    stoplist_file = options.stoplist_file

    dropout = float(options.dropout)
    init_lr = float(options.init_lr)
    patience = int(options.patience)
    max_epochs = int(options.max_epochs)

    test_prop = options.test_prop
    verbose = options.verbose

    average = 'micro'

    if do_ensemble:
        do_platt = True
        do_cfm = True
    else:
        do_platt = False
        do_cfm = False

    test_over_time(project_dir, subset, config_file, model_type, field, train_start, train_end, test_start, test_end, n_train, n_calib, penalty, suffix, loss, objective, do_ensemble, dh, label, intercept, n_dev_folds, average, seed, alpha_min, alpha_max, n_alphas, sample_labels, group_identical, annotated, nonlinearity, init_lr=init_lr, list_size=ls, repeats=repeats, oracle=oracle, lower=lower, interactive=interactive, stoplist_file=stoplist_file, cshift=cshift, n_cshift=n_cshift, do_cfm=do_cfm, do_platt=do_platt, dropout=dropout, patience=patience, max_epochs=max_epochs, test_prop=test_prop, verbose=verbose)


def test_over_time(project_dir, subset, config_file, model_type, field, train_start, train_end, test_start, test_end, n_train=None, n_calib=0, penalty='l2', suffix='', loss='log', objective='f1', do_ensemble=True, dh=300, label='label', intercept=True, n_dev_folds=5, average='micro', seed=None, alpha_min=0.01, alpha_max=1000.0, n_alphas=8, sample_labels=False, group_identical=False, annotated_subset=None, nonlinearity='tanh', init_lr=1e-2, min_epochs=2, max_epochs=50, patience=5, tol=1e-4, list_size=1, repeats=1, oracle=False, lower=None, interactive=False, stoplist_file=None, cshift=False, n_cshift=None, do_cfm=True, do_platt=True, dropout=0.0, min_test=None, test_prop=None, verbose=False):
    # Just run a regular model, one per year, training on the past, and save the reults

    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)

    log = {
        'project': project_dir,
        'subset': subset,
        'config_file': config_file,
        'model_type': model_type,
        'field': field,
        'train_start': train_start,
        'train_end': train_end,
        'test_start': test_start,
        'test_end': test_end,
        'n_train': n_train,
        'n_calib': n_calib,
        'penalty': penalty,
        'cshift': cshift,
        'n_cshift': n_cshift,
        'suffix': suffix,
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
        'nonlinearity': nonlinearity,
        'init_lr': init_lr,
        'min_epochs': min_epochs,
        'max_epochs': max_epochs,
        'patience': patience,
        'tol': tol,
        'interactive': interactive,
        'stoplist_file': stoplist_file,
        'list_size': list_size
    }

    model_basename = make_model_basename(log)

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
    field_vals = list(set(metadata[field].values))
    field_vals.sort()
    print("Splitting data according to %s", field)
    print("Values:", field_vals)

    print("\nTesting on %s to %s" % (test_start, test_end))

    # first, split into training and non-train data based on the field of interest
    all_items = list(metadata.index)
    test_selector_all = (metadata[field] >= int(test_start)) & (metadata[field] <= int(test_end))
    test_subset_all = metadata[test_selector_all]
    test_items_all = test_subset_all.index.tolist()
    n_test_all = len(test_items_all)

    if min_test is not None:
        if n_test_all < min_test:
            print("Not enough test samples; exiting")
            return

    if train_end is None:
        if train_start is None:
            train_selector_all = metadata[field] < int(test_start)
        else:
            train_selector_all = (metadata[field] < int(test_start)) & (metadata[field] >= train_start)
    else:
        if train_start is None:
            train_selector_all = metadata[field] <= int(train_end)
        else:
            train_selector_all = (metadata[field] <= int(train_end)) & (metadata[field] >= train_start)

    train_subset_all = metadata[train_selector_all]
    train_items_all = list(train_subset_all.index)
    n_train_all = len(train_items_all)
    # only keep the items in the train and test sets
    all_items = train_items_all + test_items_all

    print("Train: %d, Test: %d (labeled and unlabeled)" % (n_train_all, n_test_all))

    # load all labels
    label_dir = dirs.dir_labels(project_dir, subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
    labels_df = labels_df.loc[all_items]

    # if desired, attempt to learn weights for the training data using techniques for covariate shift
    if cshift:
        print("Training a classifier for covariate shift")
        # start by learning to discriminate train from non-train data
        # Label items based on whether they come from train or test
        train_test_labels = np.zeros((len(all_items), 2), dtype=int)
        train_test_labels[:n_train_all, 0] = 1
        train_test_labels[n_train_all:, 1] = 1
        if np.sum(train_test_labels[:, 0]) < np.sum(train_test_labels[:, 1]):
            cshift_pos_label = 0
        else:
            cshift_pos_label = 1
        train_test_labels_df = pd.DataFrame(train_test_labels, index=all_items, columns=[0, 1])

        if n_cshift is not None and len(all_items) >= n_cshift:
            print("Taking a random sample of %d items for reweighting" % n_cshift)
            #np.random.shuffle(all_items)
            cshift_items = np.random.choice(all_items, size=n_cshift, replace=False)
        else:
            print("Using all train items")
            cshift_items = all_items

        print(train_test_labels_df.loc[cshift_items].mean(axis=0))

        # create a cshift model using the same specifiction as our model below (e.g. LR/MLP, etc.)
        model_name = model_basename + '_' + str(test_start) + '-' + str(test_end) + 'cshift'
        model, dev_f1, dev_acc, dev_cal, dev_cal_overall = train.train_model_with_labels(project_dir, model_type, loss, model_name, subset, train_test_labels_df, feature_defs, items_to_use=cshift_items, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, intercept=intercept, n_dev_folds=n_dev_folds, save_model=True, do_ensemble=False, dh=dh, seed=seed, pos_label=cshift_pos_label, verbose=False)
        print("cshift results: %0.4f f1, %0.4f acc" % (dev_f1, dev_acc))

        #X_cshift, features_concat = predict.load_data(project_dir, model_name, subset, items_to_use=all_items)
        X_cshift, features_concat = predict.load_data(project_dir, model_name, subset, items_to_use=all_items)
        cshift_pred_probs = model.predict_probs(X_cshift)
        f_items = features_concat.get_items()
        assert len(f_items) == len(all_items)
        for i in range(len(all_items)):
            assert all_items[i] == f_items[i]
        cshift_pred_probs_df = pd.DataFrame(cshift_pred_probs, index=features_concat.get_items(), columns=range(2))

        # display the min and max probs
        print("Min: %0.6f" % cshift_pred_probs_df[1].values[:n_train_all].min())
        print("Mean: %0.6f" % cshift_pred_probs_df[1].values[:n_train_all].mean())
        print("Max: %0.6f" % cshift_pred_probs_df[1].values[:n_train_all].max())
        # HACK: need to prevent 0s in prob(y=0|x)
        p_train_values = cshift_pred_probs_df[0].values
        threshold = 0.01
        p_train_values[p_train_values < threshold] = threshold
        print("After thresholding")
        print("Min: %0.6f" % p_train_values[:n_train_all].min())
        print("Mean: %0.6f" % p_train_values[:n_train_all].mean())
        print("Max: %0.6f" % p_train_values[:n_train_all].max())

        # use the estimated probability of each item being a training item to compute item weights
        weights = n_train_all / float(n_test_all) * (1.0/p_train_values - 1)
        weights_df_all = pd.DataFrame(weights, index=all_items)
        # print a summary of the weights from just the training items
        print("Min weight: %0.4f" % weights[:n_train_all].min())
        print("Ave weight: %0.4f" % weights[:n_train_all].mean())
        print("Max weight: %0.4f" % weights[:n_train_all].max())
        # print a summary of all weights
        #print("Min weight: %0.4f" % weights.min())
        #print("Ave weight: %0.4f" % weights.mean())
        #print("Max weight: %0.4f" % weights.max())
        # create a data frame with this information
    else:
        weights_df_all = None

    # find the labeled items
    print("Subsetting items with labels")
    label_sums_df = labels_df.sum(axis=1)
    labeled_item_selector = label_sums_df > 0
    labels_df = labels_df[labeled_item_selector]
    n_labeled_items, n_classes = labels_df.shape
    print("%d labeled items" % n_labeled_items)
    labeled_items = set(labels_df.index)

    train_items_labeled = [i for i in train_items_all if i in labeled_items]

    test_items = [i for i in test_items_all if i in labeled_items]
    #n_train = len(train_items)
    n_test = len(test_items)

    for r in range(repeats):

        # set seed very explicily here to make sure experiments are comparable
        if seed is not None:
            seed += 1
            np.random.seed(seed)

        print("* Starting repetition %d *" % r)
        model_name = model_basename + '_' + str(test_start) + '-' + str(test_end) + '_' + str(r)
        if n_train is not None and len(train_items_labeled) >= n_train:
            np.random.shuffle(train_items_labeled)
            train_items = np.random.choice(train_items_labeled, size=n_train, replace=False)
        else:
            print("Using all train items")
            train_items = train_items_labeled
        n_train_r = len(train_items)

        # now, choose a calibration set
        if n_calib > 0 and n_test >= n_calib:
            np.random.shuffle(test_items)
            calib_items = np.random.choice(test_items, size=n_calib, replace=False)
        elif n_test < n_calib:
            print("Error: Only %d labeled test instances available" % n_test)
            calib_items = test_items
        else:
            calib_items = []

        if weights_df_all is not None:
            weights_df = weights_df_all[labeled_item_selector]
        else:
            weights_df = None

        print("Labeled train: %d, test: %d" % (n_train_r, n_test))

        # create a data frame to hold a summary of the results
        output_df = pd.DataFrame([], columns=['N', 'training data', 'test data', 'cal', 'estimate', 'MAE', '95lcl', '95ucl', 'contains_test'])

        test_labels_df = labels_df.loc[test_items]
        # do a fake adjustment of the test label proportions
        if test_prop is not None:
            test_prop = float(test_prop)
            test_label_values = test_labels_df.values
            test_label_props = test_label_values[:, 1] / (test_label_values[:, 1] + test_label_values[:, 0])
            order = list(np.argsort(test_label_props))

            true_prop = np.mean(test_label_props)
            if test_prop < true_prop:
                i = 0
                running = test_label_props[order[i]]
                new_test_items = [test_items[order[i]]]
                i += 1
                while (running / i) <= test_prop:
                    running += test_label_props[order[i]]
                    new_test_items.append(test_items[order[i]])
                    i += 1
                print("Taking %d test_items" % len(new_test_items))
                test_items = new_test_items[:]
            else:
                order.reverse()
                i = 0
                running = test_label_props[order[i]]
                new_test_items = [test_items[order[i]]]
                i += 1
                while (running / i) >= test_prop:
                    running += test_label_props[order[i]]
                    new_test_items.append(test_items[order[i]])
                    i += 1
                print("Taking %d test_items" % len(new_test_items))
                test_items = new_test_items[:]

            test_labels_df = labels_df.loc[test_items]
            test_label_values = test_labels_df.values
            test_label_props = test_label_values[:, 1] / (test_label_values[:, 1] + test_label_values[:, 0])
            print("New props = %0.3f" % np.mean(test_label_props))

        # if instructed, sample labels in proportion to annotations (to simulate having one label per item)
        if sample_labels:
            print("Sampling labels")
            # normalize the labels
            temp = labels_df.values / np.array(labels_df.values.sum(axis=1).reshape((n_labeled_items, 1)), dtype=float)
            samples = np.zeros([n_labeled_items, n_classes], dtype=int)
            for i in range(n_labeled_items):
                index = np.random.choice(np.arange(n_classes), size=1, p=temp[i, :])
                samples[i, index] = 1
            sampled_labels_df = pd.DataFrame(samples, index=labels_df.index, columns=labels_df.columns)
        else:
            sampled_labels_df = labels_df

        train_labels_df = sampled_labels_df.loc[train_items].copy()
        if n_calib > 0:
            calib_labels_df = sampled_labels_df.loc[calib_items].copy()
        else:
            calib_labels_df = None

        # get the true proportion of labels in the test OR non-training data (calibration and test combined)
        target_props, target_estimate, target_std = get_estimate_and_std(test_labels_df, use_n_annotations=True)
        output_df.loc['target'] = [n_test, 'test', 'test', 'n/a', target_estimate, 0, target_estimate - 2 * target_std, target_estimate + 2 * target_std, np.nan]

        # get the same estimate from training data
        train_props, train_estimate, train_std = get_estimate_and_std(train_labels_df, use_n_annotations=True)
        print("Train props:", train_props, train_estimate)
        train_rmse = np.abs(train_estimate - target_estimate)
        train_contains_test = target_estimate > train_estimate - 2 * train_std and target_estimate < train_estimate + 2 * train_std
        output_df.loc['train'] = [n_train_r, 'train', 'test', 'n/a', train_estimate, train_rmse, train_estimate - 2 * train_std, train_estimate + 2 * train_std, train_contains_test]

        # get the same estimate from training data
        if n_calib > 0:
            calib_props, calib_estimate, calib_std = get_estimate_and_std(calib_labels_df, use_n_annotations=True)
            # compute the error of this estimate
            calib_rmse = np.abs(calib_estimate - target_estimate)
            calib_contains_test = target_estimate > calib_estimate - 2 * calib_std and target_estimate < calib_estimate + 2 * calib_std
            output_df.loc['calib'] = [n_calib, 'calib', 'test', 'n/a', calib_estimate, calib_rmse, calib_estimate - 2 * calib_std, calib_estimate + 2 * calib_std, calib_contains_test]
        else:
            calib_estimate = 0.0
            calib_std = 1.0
            output_df.loc['calib'] = [n_calib, 'calib', 'test', 'n/a', np.nan, np.nan, np.nan, np.nan, np.nan]

        if train_estimate > 0.5:
            pos_label = 0
        else:
            pos_label = 1
        print("Using %d as the positive label" % pos_label)

        results_df = pd.DataFrame([], columns=['f1', 'acc', 'mae', 'estimated calibration'])

        # Now train a model on the training data, saving the calibration data for calibration

        if stoplist_file is not None:
            stoplist = fh.read_text(stoplist_file)
            stoplist = {s.strip() for s in stoplist}
            print(stoplist)
        else:
            stoplist = None

        print("Training a LR model")
        model, dev_f1, dev_acc, dev_cal_mae, dev_cal_est = train.train_model_with_labels(project_dir, model_type, 'log', model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, intercept=intercept, objective=objective, n_dev_folds=n_dev_folds, do_ensemble=do_ensemble, dh=dh, seed=seed, pos_label=pos_label, vocab=None, group_identical=group_identical, nonlinearity=nonlinearity, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, do_cfm=do_cfm, do_platt=do_platt, lower=lower, stoplist=stoplist, dropout=dropout, verbose=verbose)
        results_df.loc['cross_val'] = [dev_f1, dev_acc, dev_cal_mae, dev_cal_est]

        X_test, features_concat = predict.load_data(project_dir, model_name, subset, items_to_use=test_items)
        test_predictions = model.predict(X_test)
        test_predictions_df = pd.DataFrame(test_predictions, index=features_concat.get_items(), columns=[label])
        test_pred_probs = model.predict_probs(X_test)
        _, n_labels = test_pred_probs.shape
        test_pred_probs_df = pd.DataFrame(test_pred_probs, index=features_concat.get_items(), columns=range(n_labels))

        f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
        true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
        test_cal_est = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
        test_cc_estimate, test_pcc_estimate = model.predict_proportions(X_test)

        test_cc_mae = np.mean(np.abs(test_cc_estimate[1] - target_estimate))
        test_pcc_mae = np.mean(np.abs(test_pcc_estimate[1] - target_estimate))

        results_df.loc['test'] = [f1_test, acc_test, test_pcc_mae, test_cal_est]

        output_df.loc['CC'] = [n_train_r, 'train', 'test', 'n/a', test_cc_estimate[1], test_cc_mae, np.nan, np.nan, np.nan]
        output_df.loc['PCC'] = [n_train_r, 'train', 'test', 'n/a', test_pcc_estimate[1], test_pcc_mae, np.nan, np.nan, np.nan]

        test_acc_estimate_internal, test_acc_ms_estimate_internal = model.predict_proportions(X_test, do_cfm=do_cfm)

        test_acc_rmse_internal = np.abs(test_acc_estimate_internal[1] - target_estimate)
        test_acc_ms_rmse_internal = np.abs(test_acc_ms_estimate_internal[1] - target_estimate)

        output_df.loc['ACC_internal'] = [n_train_r, 'train', 'test', 'n/a', test_acc_estimate_internal[1], test_acc_rmse_internal, np.nan, np.nan, np.nan]
        output_df.loc['MS_internal'] = [n_train_r, 'train', 'nontrain', 'predicted', test_acc_ms_estimate_internal[1], test_acc_ms_rmse_internal, np.nan, np.nan, np.nan]

        test_platt1_estimate, test_platt2_estimate = model.predict_proportions(X_test, do_platt=do_platt)

        test_platt1_rmse = np.abs(test_platt1_estimate[1] - target_estimate)
        test_platt2_rmse = np.abs(test_platt2_estimate[1] - target_estimate)

        output_df.loc['PCC_platt1'] = [n_train_r, 'train', 'test', 'n/a', test_platt1_estimate[1], test_platt1_rmse, np.nan, np.nan, np.nan]
        output_df.loc['PCC_platt2'] = [n_train_r, 'train', 'nontrain', 'predicted', test_platt2_estimate[1], test_platt2_rmse, np.nan, np.nan, np.nan]

        if n_calib > 0:
            cc_plus_cal_estimate = (test_cc_estimate[1] + calib_estimate) / 2.0
            pcc_plus_cal_estimate = (test_pcc_estimate[1] + calib_estimate) / 2.0
            cc_plus_cal_mae = np.mean(np.abs(cc_plus_cal_estimate - target_estimate))
            pcc_plus_cal_mae = np.mean(np.abs(pcc_plus_cal_estimate - target_estimate))

            #output_df.loc['CC_plus_cal'] = [n_train, 'train', 'test', 'n/a', cc_plus_cal_estimate, cc_plus_cal_mae, np.nan, np.nan, np.nan]
            output_df.loc['PCC_plus_cal'] = [n_train_r, 'train', 'test', 'n/a', pcc_plus_cal_estimate, pcc_plus_cal_mae, np.nan, np.nan, np.nan]

        results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))
        output_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))

        """
        print("Training a DL model")        
        feature_list = None
        if oracle:
            feature_list = []
            oracle_name = model_name + "_DL_" + str(list_size) + '_oracle'
            ensemble, dev_f1, dev_acc, dev_cal_mae, dev_cal_est = train.train_model_with_labels(project_dir, 'DL', 'log', oracle_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=test_items, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, intercept=intercept, objective=objective, n_dev_folds=2, do_ensemble=True, dh=dh, seed=seed, pos_label=pos_label, vocab=None, group_identical=group_identical, nonlinearity=nonlinearity, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, tol=tol, early_stopping=early_stopping, list_size=1, lower=lower, verbose=verbose)
            for name, model in ensemble._models.items():
                feature_list.extend(model.get_feature_list())

        model_name = model_name + "_DL_" + str(list_size)
        model, dev_f1, dev_acc, dev_cal_mae, dev_cal_est = train.train_model_with_labels(project_dir, 'DL', 'log', model_name, subset, sampled_labels_df, feature_defs, weights_df=weights_df, items_to_use=train_items, penalty=penalty, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, intercept=intercept, objective=objective, n_dev_folds=2, do_ensemble=True, dh=dh, seed=seed, pos_label=pos_label, vocab=None, group_identical=group_identical, nonlinearity=nonlinearity, init_lr=init_lr, min_epochs=min_epochs, max_epochs=max_epochs, patience=patience, tol=tol, early_stopping=early_stopping, list_size=list_size, dl_feature_list=feature_list, interactive=interactive, verbose=verbose)
        results_df.loc['cross_val'] = [dev_f1, dev_acc, dev_cal_mae, dev_cal_est]

        # predict on test data
        force_dense = False

        X_test, features_concat = predict.load_data(project_dir, model_name, subset, items_to_use=test_items, force_dense=force_dense)

        predict.test_DL_model(project_dir, model, subset, label, X_test, items_to_use=test_items)

        test_predictions = model.predict(X_test)
        test_predictions_df = pd.DataFrame(test_predictions, index=features_concat.get_items(), columns=[label])
        test_pred_probs = model.predict_probs(X_test)
        _, n_labels = test_pred_probs.shape
        test_pred_probs_df = pd.DataFrame(test_pred_probs, index=features_concat.get_items(), columns=range(n_labels))

        #test_predictions_df, test_pred_probs_df, test_pred_proportions, samples = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose, force_dense=force_dense, group_identical=group_identical, n_samples=100)
        f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
        true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
        test_cal_est = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
        #test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_pvc_estimate_internal = test_pred_proportions

        test_cc_estimate, test_pcc_estimate = model.predict_proportions(X_test)

        test_cc_mae = np.mean(np.abs(test_cc_estimate[1] - target_estimate))
        test_pcc_mae = np.mean(np.abs(test_pcc_estimate[1] - target_estimate))

        results_df.loc['test'] = [f1_test, acc_test, test_pcc_mae, test_cal_est]

        output_df.loc['CC_DL'] = [n_train_r, 'train', 'test', 'n/a', test_cc_estimate[1], test_cc_mae, np.nan, np.nan, np.nan]
        output_df.loc['PCC_DL'] = [n_train_r, 'train', 'test', 'n/a', test_pcc_estimate[1], test_pcc_mae, np.nan, np.nan, np.nan]

        _, test_pcc_estimate_platt = model.predict_proportions(X_test, do_platt=True)
        test_pcc_platt_mae = np.mean(np.abs(test_pcc_estimate_platt [1] - target_estimate))
        output_df.loc['PCC_DL_platt2'] = [n_train_r, 'train', 'test', 'n/a', test_pcc_estimate_platt[1], test_pcc_platt_mae, np.nan, np.nan, np.nan]

        if n_calib > 0:
            cc_plus_cal_estimate = (test_cc_estimate[1] + calib_estimate) / 2.0
            pcc_plus_cal_estimate = (test_pcc_estimate[1] + calib_estimate) / 2.0
            pcc_plus_cal_mae = np.mean(np.abs(pcc_plus_cal_estimate - target_estimate))

            #output_df.loc['CC_plus_cal'] = [n_train, 'train', 'test', 'n/a', cc_plus_cal_estimate, cc_plus_cal_mae, np.nan, np.nan, np.nan]
            output_df.loc['PCC_DL_plus_cal'] = [n_train_r, 'train', 'test', 'n/a', pcc_plus_cal_estimate, pcc_plus_cal_mae, np.nan, np.nan, np.nan]

        #samples = predict.sample_predictions(model, X_test, n_samples=100)
        #pcc_samples = np.mean(samples, axis=0)
        #sample_pcc = np.mean(pcc_samples)
        #sample_pcc_lower = np.percentile(pcc_samples, q=2.5)
        #sample_pcc_upper = np.percentile(pcc_samples, q=97.5)
        #sample_pcc_var = np.var(pcc_samples)
        #sample_pcc_mae = np.mean(np.abs(sample_pcc - target_estimate))
        #sample_pcc_contains_test = target_estimate > sample_pcc_lower and target_estimate < sample_pcc_upper
        #output_df.loc['PCC_samples'] = [n_train_r, 'train', 'test', 'n/a', sample_pcc, sample_pcc_mae, sample_pcc_lower, sample_pcc_upper, sample_pcc_contains_test]
        output_df.loc['PCC_samples'] = [n_train_r, 'train', 'test', 'n/a', np.nan, np.nan, np.nan, np.nan, np.nan]

        if n_calib > 0:
            #pcc_plus_cal_estimate = (sample_pcc / sample_pcc_var + calib_estimate / calib_std ** 2) / (1.0 / sample_pcc_var + 1.0 / calib_std ** 2)
            #pcc_plus_cal_mae = np.mean(np.abs(pcc_plus_cal_estimate - target_estimate))
            #pcc_plus_cal_std = np.sqrt(1.0 / (1.0 / sample_pcc_var + 1.0 / calib_std ** 2))
            #pcc_plus_cal_contains_test = target_estimate > pcc_plus_cal_estimate - 2 * pcc_plus_cal_std and target_estimate < pcc_plus_cal_estimate + 2 * pcc_plus_cal_std
            #output_df.loc['PCC_samples_plus_cal'] = [n_train_r, 'train', 'test', 'n/a', pcc_plus_cal_estimate, pcc_plus_cal_mae, pcc_plus_cal_estimate - 2 * pcc_plus_cal_std, pcc_plus_cal_estimate + 2 * pcc_plus_cal_std, pcc_plus_cal_contains_test]
            output_df.loc['PCC_samples_plus_cal'] = [n_train_r, 'train', 'test', 'n/a', np.nan, np.nan, np.nan, np.nan, np.nan]
        else:
            output_df.loc['PCC_samples_plus_cal'] = [n_train_r, 'train', 'test', 'n/a', np.nan, np.nan, np.nan, np.nan, np.nan]

        results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))
        output_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))
        """


def make_model_basename(log):
    model_basename = log['subset'] + '_' + log['label'] + '_' + log['penalty'] + '_' + log['objective']
    if log['n_train'] is None:
        model_basename += '_0'
    else:
        model_basename += '_' + str(log['n_train'])
    model_basename += '_' + str(log['n_calib'])
    if log['model_type'] == 'DAN':
        model_basename += '_dh' + str(log['dh'])
    if log['sample_labels']:
        model_basename += '_sampled'
    if log['cshift']:
        model_basename += '_cshift'
    model_basename += log['suffix']
    return model_basename


def get_estimate_and_std(labels_df, use_n_annotations=False):
    n_items, n_classes = labels_df.shape
    assert n_classes == 2
    labels = labels_df.values.copy()

    if use_n_annotations:
        # treat each annotation as a separate sample
        n = np.sum(labels)
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



"""
# An negative exploration of adding random weights for previously-unseen features

# Try predicting with expanded model
X_test, features_concat, new_indices = predict.load_data_expanded_model(project_dir, model_name, subset, items_to_use=test_items, force_dense=force_dense, verbose=True)
print("Loaded test data: (%d, %d)" % X_test.shape)

# update model with expanded coefs
print("Updating model")
assert model.get_model_type() == 'LR'
old_coefs = model.get_coef_array()
print(old_coefs.shape)
old_terms = model._col_names
n_classes, n_terms_old = old_coefs.shape

new_terms = features_concat.terms
n_terms_new = len(new_terms)
new_index = dict(zip(new_terms, np.arange(n_terms_new)))
print("Creating new coefficient matrix")
print(n_terms_new, n_classes)

# Creating a copy of self.counts with an extra column of zeros
zeros_col = np.zeros([n_classes, 1])
temp = np.hstack([old_coefs, zeros_col])

index = np.ones(n_terms_new, dtype=int) * n_terms_old
for t_i, term in enumerate(old_terms):
    index[new_index[term]] = t_i
new_coefs = temp[:, index]
print(new_coefs.shape)
model.set_coefs(new_terms, new_coefs)

test_predictions = model.predict(X_test)
test_predictions_df = pd.DataFrame(test_predictions, index=features_concat.get_items(), columns=[label])
test_pred_probs = model.predict_probs(X_test)
_, n_labels = test_pred_probs.shape
test_pred_probs_df = pd.DataFrame(test_pred_probs, index=features_concat.get_items(), columns=range(n_labels))

#test_predictions_df, test_pred_probs_df, test_pred_proportions, _ = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose, force_dense=force_dense)
f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average)
true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
#test_cal_mae = evaluation.eval_proportions_mae(test_labels_df.as_matrix(), test_pred_probs_df.as_matrix())
test_cal_est = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
#test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_acc_ms_estimate_internal = test_pred_proportions

test_cc_estimate, test_pcc_estimate = model.predict_proportions(X_test)

test_cc_mae = np.mean(np.abs(test_cc_estimate[1] - target_estimate))
test_pcc_mae = np.mean(np.abs(test_pcc_estimate[1] - target_estimate))

results_df.loc['test_expanded'] = [f1_test, acc_test, test_pcc_mae, test_cal_est]

output_df.loc['CC_expanded'] = [n_train_r, 'train', 'test', 'n/a', test_cc_estimate[1], test_cc_mae, np.nan, np.nan, np.nan]
output_df.loc['PCC_expanded'] = [n_train_r, 'train', 'test', 'n/a', test_pcc_estimate[1], test_pcc_mae, np.nan, np.nan, np.nan]

n_new_features_per_item = X_test[:, new_indices].sum(axis=1)
familiar_subset = n_new_features_per_item == 0
print("Num familiar documents = %d" % np.sum(familiar_subset))

# try sampling random weights and looking at the variation.
print("Trying sampling")
cc_estimates = []
pcc_estimates = []
for s in range(500):
    # start by sampling coefficients from the empirical distribution of the original model
    new_coefs = np.random.choice(old_coefs.reshape((n_classes * n_terms_old, )), size=(n_classes, n_terms_new), replace=True)
    # then copy over the coefficients from the old model
    for t_i, term in enumerate(old_terms):
        new_coefs[:, new_index[term]] = old_coefs[:, t_i]
    # assign the new coefficients to the model
    model.set_coefs(new_terms, new_coefs)

    # repeat the prediction process
    test_predictions = model.predict(X_test)
    test_predictions_df = pd.DataFrame(test_predictions, index=features_concat.get_items(), columns=[label])
    test_pred_probs = model.predict_probs(X_test)
    _, n_labels = test_pred_probs.shape
    test_pred_probs_df = pd.DataFrame(test_pred_probs, index=features_concat.get_items(), columns=range(n_labels))

    #test_predictions_df, test_pred_probs_df, test_pred_proportions, _ = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items, verbose=verbose, force_dense=force_dense)
    f1_test, acc_test = evaluate_predictions.evaluate_predictions(test_labels_df, test_predictions_df, test_pred_probs_df, pos_label=pos_label, average=average, verbose=False)
    true_test_vector = np.argmax(test_labels_df.as_matrix(), axis=1)
    #test_cal_mae = evaluation.eval_proportions_mae(test_labels_df.as_matrix(), test_pred_probs_df.as_matrix())
    test_cal_est = evaluation.evaluate_calibration_rmse(true_test_vector, test_pred_probs_df.as_matrix(), min_bins=1, max_bins=1)
    #test_cc_estimate, test_pcc_estimate, test_acc_estimate_internal, test_acc_ms_estimate_internal = test_pred_proportions

    test_cc_estimate, test_pcc_estimate = model.predict_proportions(X_test)
    cc_estimates.append(test_cc_estimate[1])
    pcc_estimates.append(test_pcc_estimate[1])

cc_mean = np.mean(cc_estimates)
pcc_mean = np.mean(pcc_estimates)

test_cc_mae = np.mean(np.abs(cc_mean - target_estimate))
test_pcc_mae = np.mean(np.abs(pcc_mean - target_estimate))

cc_lower = np.percentile(cc_estimates, q=2.5)
cc_upper = np.percentile(cc_estimates, q=97.5)
cc_contains_test = target_estimate > cc_lower and target_estimate < cc_upper
output_df.loc['CC_expanded_sample'] = [n_train_r, 'train', 'test', 'n/a', cc_mean, test_cc_mae, cc_lower, cc_upper, cc_contains_test]

pcc_lower = np.percentile(pcc_estimates, q=2.5)
pcc_upper = np.percentile(pcc_estimates, q=97.5)
pcc_contains_test = target_estimate > pcc_lower and target_estimate < pcc_upper
output_df.loc['PCC_expanded_sample'] = [n_train_r, 'train', 'test', 'n/a', pcc_mean, test_pcc_mae, pcc_lower, pcc_upper, pcc_contains_test]

results_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'accuracy.csv'))
output_df.to_csv(os.path.join(dirs.dir_models(project_dir), model_name, 'results.csv'))

"""