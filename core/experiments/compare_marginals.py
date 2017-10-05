import os
import re
import sys
from optparse import OptionParser
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
from scipy import sparse
from scipy import stats
from sklearn.model_selection import KFold

from ..util import file_handling as fh
from ..models import linear, mlp, evaluation, calibration, ensemble
from ..preprocessing import features
from ..util import dirs
from ..util.misc import printv

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():
    usage = "%prog project_dir subset config.json field_name"
    parser = OptionParser(usage=usage)
    #parser.add_option('--model', dest='model', default='LR',
    #                  help='Model type [LR|MLP]: default=%default')
    #parser.add_option('--loss', dest='loss', default='log',
    #                  help='Loss function [log|brier]: default=%default')
    #parser.add_option('--dh', dest='dh', default=0,
    #                  help='Hidden layer size for MLP [0 for None]: default=%default')
    #parser.add_option('--ensemble', action="store_true", dest="ensemble", default=False,
    #                  help='Make an ensemble from cross-validation, instead of training one model: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    #parser.add_option('--weights', dest='weights_file', default=None,
    #                  help='Weights file: default=%default')
    #parser.add_option('--penalty', dest='penalty', default='l1',
    #                  help='Regularization type: default=%default')
    #parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
    #                  help='Use to fit a model with no intercept: default=%default')
    #parser.add_option('--objective', dest='objective', default='f1',
    #                  help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    config_file = args[2]
    field_name = args[3]

    label = options.label
    n_dev_folds = int(options.n_dev_folds)
    seed = options.seed
    if seed is not None:
        seed = int(seed)
        np.random.seed(seed)

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    compare_marginals(project_dir, subset, label, field_name, feature_defs, n_dev_folds=n_dev_folds, seed=seed)


def compare_marginals(project_dir, subset, label, field_name, feature_defs, items_to_use=None, n_dev_folds=5, seed=None, verbose=True):

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

        non_train_selector = metadata[field_name] == v
        non_train_subset = metadata[non_train_selector]
        non_train_items = non_train_subset.index.tolist()

        # load all labels
        label_dir = dirs.dir_labels(project_dir, subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)
        n_items, n_classes = labels_df.shape


        features_dir = dirs.dir_features(project_dir, subset)
        n_items, n_classes = labels_df.shape

        printv("loading training features", verbose)
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
            indices_to_use = [feature_item_index[i] for i in train_items]
            if indices_to_use is not None:
                printv("Taking subset of items", verbose)
                feature = features.create_from_feature(feature, indices_to_use)
                printv("New shape = (%d, %d)" % feature.get_shape(), verbose)
            feature.threshold(feature_def.min_df)
            if feature_def.transform == 'doc2vec':
                word_vectors_prefix = os.path.join(features_dir, name + '_vecs')
            else:
                word_vectors_prefix = None
            feature.transform(feature_def.transform, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
            printv("Final shape = (%d, %d)" % feature.get_shape(), verbose)
            feature_list.append(feature)
            feature_signature = features.get_feature_signature(feature_def, feature)
            feature_signature['word_vectors_prefix'] = word_vectors_prefix
            feature_signatures.append(feature_signature)

        #output_dir = os.path.join(dirs.dir_models(project_dir), model_name)
        #if save_model:
        #    fh.makedirs(output_dir)
        #    fh.write_to_json(feature_signatures, os.path.join(output_dir, 'features.json'), sort_keys=False)

        features_concat = features.concatenate(feature_list)
        col_names = features_concat.get_col_names()

        if features_concat.sparse:
            X_train = features_concat.get_counts().tocsr()
        else:
            X_train = features_concat.get_counts()
        Y_train = labels_df.as_matrix()

        n_train, n_features = X_train.shape
        _, n_classes = Y_train.shape

        #index = col_names.index(target_word)
        #if index < 0:
        #    sys.exit("word not found in feature")

        test_features_dir = dirs.dir_features(project_dir, subset)
        printv("loading test features", verbose)
        items_to_use = non_train_items
        feature_list = []
        for sig in feature_signatures:
            feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
            printv("Loading %s" % feature_def, verbose)
            name = feature_def.name
            test_feature = features.load_from_file(input_dir=test_features_dir, basename=name)
            printv("Initial shape = (%d, %d)" % test_feature.get_shape(), verbose)

            # use only a subset of the items, if given
            if items_to_use is not None:
                all_test_items = test_feature.get_items()
                n_items = len(all_test_items)
                item_index = dict(zip(all_test_items, range(n_items)))
                indices_to_use = [item_index[i] for i in items_to_use]
                printv("Taking subset of items", verbose)
                test_feature = features.create_from_feature(test_feature, indices_to_use)
                printv("New shape = (%d, %d)" % test_feature.get_shape(), verbose)
            printv("Setting vocabulary", verbose)
            test_feature.set_terms(sig['terms'])
            idf = None
            if feature_def.transform == 'tfidf':
                idf = sig['idf']
            word_vectors_prefix = sig['word_vectors_prefix']
            test_feature.transform(feature_def.transform, idf=idf, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
            printv("Final shape = (%d, %d)" % test_feature.get_shape(), verbose)
            feature_list.append(test_feature)
            #output_dir = os.path.join(dirs.dir_models(project_dir), model_name)
            #if save_model:
            #    fh.makedirs(output_dir)
            #    fh.write_to_json(feature_signatures, os.path.join(output_dir, 'features.json'), sort_keys=False)

        features_concat = features.concatenate(feature_list)

        if features_concat.sparse:
            X_nontrain = features_concat.get_counts().tocsr()
        else:
            X_nontrain = features_concat.get_counts()
        Y_nontrain = labels_df.as_matrix()

        n_nontrain, _ = X_nontrain.shape

        target_words = ['amendment', 'attorney', 'ban', 'benefits', 'case', 'civil', 'constitution', 'constitutional', 'court', 'courts', 'decision', 'decisions', 'federal', 'filed', 'granted', 'judge', 'judges', 'judicial', 'law', 'laws', 'lawsuit', 'lawyer', 'lawyers', 'legal', 'legalized', 'legality', 'licenses', 'majority', 'order', 'prop', 'protections', 'right', 'ruled', 'ruling', 'senate', 'suit', 'sued', 'supreme', 'unconstitutional']
        #target_words = ['court', 'law', 'judge', 'legal']
        print("n words = ", len(target_words))

        indices = [col_names.index(w) for w in target_words]
        print(indices)

        train_counts = defaultdict(int)
        positives = defaultdict(list)
        for i in range(n_train):
            if np.sum(Y_train[i, :]) > 0:
                ps_i = Y_train[i, :] / np.sum(Y_train[i, :])
                p = ps_i[1]
                vector = np.array(X_train[i, indices].todense()).ravel()
                key = ''.join([str(int(s)) for s in vector])
                train_counts[key] += 1
                positives[key].append(p)

        print(len(train_counts), np.sum(list(train_counts.values())))

        train_keys = list(train_counts.keys())
        train_keys.sort()
        key_sums = defaultdict(int)
        for key in train_keys:
            key_sums[key] = np.sum([int(w) for w in key])
        for key in train_keys[:20]:
            print(key, train_counts[key], np.mean(positives[key]), key_sums[key])

        nontrain_counts = defaultdict(int)
        for i in range(n_nontrain):
            vector = np.array(X_nontrain[i, indices].todense()).ravel()
            key = ''.join([str(int(s)) for s in vector])
            nontrain_counts[key] += 1

        print(len(nontrain_counts), np.sum(list(nontrain_counts.values())))

        matching_lower = []
        matching_counts = []
        matching_upper = []
        keys = list(nontrain_counts.keys())
        keys.sort()
        for key in keys[1:]:
            matching_counts.append(train_counts[key])

            key_sum = key_sums[key]
            pattern = re.sub('1', '[0-1]', key)
            #print(key, pattern)
            matches = [key for key in train_keys if re.match(pattern, key) is not None and key_sums[key] > 0]
            #print(matches)
            values = [train_counts[key] for key in matches]
            #print(values)
            count = sum(values)
            #print(sum)
            #lower = [key for key in keys if key_sum - key_sums[key] < 3 and key_sums[key] > 0]
            matching_lower.append(int(count))

            pattern = re.sub('0', '[0-1]', key)
            matches = [key for key in train_keys if re.match(pattern, key) is not None and 0 < key_sums[key] - key_sum < 10]
            values = [train_counts[key] for key in matches]
            count = sum(values)
            #lower = [key for key in keys if key_sum - key_sums[key] < 3 and key_sums[key] > 0]
            matching_upper.append(int(count))

        print(np.histogram(matching_counts))
        print(sum([count == 0 for count in matching_counts]))
        print(np.histogram(matching_lower))
        print(sum([count == 0 for count in matching_lower]))
        print(np.histogram(matching_upper))
        print(sum([count == 0 for count in matching_upper]))


def prepare_data(X, Y, weights=None, predictions=None, loss='log'):
    """
    Expand the feature matrix and label matrix by converting items with multiple labels to multiple rows with 1 each
    :param X: feature matrix (n_items, n_features)
    :param Y: label matrix (n_items, n_classes)
    :param weights: (n_items, )
    :param predictions: optional vector of predcitions to expand in parallel
    :param loss: loss function (determines whether to expand or average
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
            # otherwise, duplicate items with all labels and weights
            for index, count in enumerate(labels):
                # if there is at least one annotation for this class
                if count > 0:
                    # create a row representing an annotation for this class
                    X_list.append(X[i, :])
                    label_vector = np.zeros(n_classes, dtype=int)
                    label_vector[index] = 1
                    Y_list.append(label_vector)
                    # give it a weight based on prior weighting and the proportion of annotations for this class
                    weights_list.append(weights[i] * count/total)
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

    elif loss == 'brier':
        Y_list = []
        # just normalize labels
        for i in range(n_items):
            labels = Y[i, :]
            Y_list.append(labels / float(labels.sum()))
        X_return = X.copy()
        Y_return = np.array(Y_list)
        if weights is None:
            weights_return = np.ones(n_items)
        else:
            weights_return = np.array(weights)
        if predictions is not None:
            pred_return = np.array(predictions)
    else:
        sys.exit("Loss %s not recognized" % loss)

    if predictions is None:
        return X_return, Y_return, weights_return
    else:
        return X_return, Y_return, weights_return, pred_return


def fit_beta(values):
    mean = np.mean(values)
    var = np.var(values)
    assert var < mean * (1 - mean)
    common = mean * (1 - mean) / var - 1
    alpha = mean * common
    beta = (1 - mean) * common
    return alpha, beta


def fit_beta2(values):
    sample_size = len(values)
    mean = np.mean(values)
    var = np.var(values)
    alpha = mean * sample_size
    beta = (1 - mean) * sample_size
    return alpha, beta

if __name__ == '__main__':
    main()
