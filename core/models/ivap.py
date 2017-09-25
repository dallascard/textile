import os
import time
import bisect
from optparse import OptionParser

from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression

from ..models import isotonic_regression
from ..preprocessing import features
from ..util.misc import printv
from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()


def estimate_probs_from_labels_internal(project_dir, full_model, model_name, test_subset, test_items=None, verbose=False, plot=False):

    # do IVAP assuming that the dev scores to use are the ones that have been saved with the model


    #n_items, n_classes = labels_df.shape

    #if calib_items is not None:
    #    labels_df = labels_df.loc[calib_items]
    #    n_items, n_classes = labels_df.shape

    # normalize labels to just count one each
    #labels = labels_df.values.copy()

    # weight examples by the total number of votes they have received
    #calib_weights = labels.sum(axis=1).copy()

    #labels = labels / np.reshape(labels.sum(axis=1), (n_items, 1))

    #if weights_df is not None and calib_items is not None:
    #    calib_weights = np.array(weights_df.loc[calib_items].values).reshape((n_items,))
    #else:
    #    calib_weights = np.ones(n_items)

    model_dir = os.path.join(dirs.dir_models(project_dir), model_name)

    feature_signatures = fh.read_json(os.path.join(model_dir, 'features.json'))
    #calib_features_dir = dirs.dir_features(project_dir, calib_subset)

    """
    printv("Loading features", verbose)
    calib_feature_list = []
    for sig in feature_signatures:
        feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
        printv("Loading %s" % feature_def, verbose)
        name = feature_def.name
        calib_feature = features.load_from_file(input_dir=calib_features_dir, basename=name)
        printv("Initial shape = (%d, %d)" % calib_feature.get_shape(), verbose)

        # use only a subset of the items, if given
        if calib_items is not None:
            all_test_items = calib_feature.get_items()
            n_items = len(all_test_items)
            item_index = dict(zip(all_test_items, range(n_items)))
            indices_to_use = [item_index[i] for i in calib_items]
            printv("Taking subset of items", verbose)
            calib_feature = features.create_from_feature(calib_feature, indices_to_use)
            printv("New shape = (%d, %d)" % calib_feature.get_shape(), verbose)
        printv("Setting vocabulary", verbose)
        calib_feature.set_terms(sig['terms'])
        idf = None
        if feature_def.transform == 'tfidf':
            idf = sig['idf']
        word_vectors_prefix = sig['word_vectors_prefix']
        calib_feature.transform(feature_def.transform, idf=idf, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
        printv("Final shape = (%d, %d)" % calib_feature.get_shape(), verbose)
        calib_feature_list.append(calib_feature)

    features_concat = features.concatenate(calib_feature_list)
    if features_concat.sparse:
        calib_X = features_concat.get_counts().tocsr()
    else:
        calib_X = features_concat.get_counts()

    calib_y = labels[:, 1]

    printv("Feature matrix shape: (%d, %d)" % calib_X.shape, verbose)

    calib_pred_probs = model.predict_probs(calib_X)
    n_calib, n_classes = calib_pred_probs.shape
    assert n_classes == 2

    calib_scores = calib_pred_probs[:, 1]
    """

    test_features_dir = dirs.dir_features(project_dir, test_subset)

    printv("Loading features", verbose)
    test_feature_list = []
    for sig in feature_signatures:
        feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
        printv("Loading %s" % feature_def, verbose)
        name = feature_def.name
        test_feature = features.load_from_file(input_dir=test_features_dir, basename=name)
        printv("Initial shape = (%d, %d)" % test_feature.get_shape(), verbose)

        # use only a subset of the items, if given
        if test_items is not None:
            all_test_items = test_feature.get_items()
            n_items = len(all_test_items)
            item_index = dict(zip(all_test_items, range(n_items)))
            indices_to_use = [item_index[i] for i in test_items]
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
        test_feature_list.append(test_feature)

    features_concat = features.concatenate(test_feature_list)
    if features_concat.sparse:
        test_X = features_concat.get_counts().tocsr()
    else:
        test_X = features_concat.get_counts()

    n_test, _ = test_X.shape
    printv("Feature matrix shape: (%d, %d)" % test_X.shape, verbose)

    if full_model._model_type == 'ensemble':
        n_items, _ = test_X.shape
        n_models = full_model.get_n_models()
        test_pred_ranges_all = np.zeros([n_models, n_items, 2])
        for model_i, model_name in enumerate(full_model._models.keys()):
            test_pred_ranges_all[model_i, :, :] = get_pred_for_one_model_internal(full_model._models[model_name], test_X, plot=plot)

        geometric_means = np.zeros([n_items, 2])
        for i in range(n_items):
            geometric_means[i, 0] = np.exp(np.sum(np.log(1.0 - test_pred_ranges_all[:, i, 0])) / float(n_models))
            geometric_means[i, 1] = np.exp(np.sum(np.log(test_pred_ranges_all[:, i, 1])) / float(n_models))

        test_pred_ranges = np.zeros([n_items, 2])
        #test_pred_ranges[:, 1] = np.max(test_pred_ranges_all[:, :, 1], axis=0)
        #test_pred_ranges[:, 0] = np.min(test_pred_ranges_all[:, :, 0], axis=0)
        #test_pred_ranges[:, 0] = 1.0 - geometric_means[:, 0]
        #test_pred_ranges[:, 1] = geometric_means[:, 1]
        test_pred_ranges[:, 1] = np.mean(test_pred_ranges_all[:, :, 1], axis=0)
        test_pred_ranges[:, 0] = np.mean(test_pred_ranges_all[:, :, 0], axis=0)

        combo = geometric_means[:, 1] / (geometric_means[:, 0] + geometric_means[:, 1])

    else:
        test_pred_ranges = get_pred_for_one_model_internal(full_model, test_X, plot=plot)
        combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])

    return test_pred_ranges, combo


def get_pred_for_one_model_internal(model, test_X, plot=False):
    test_scores = model.predict_probs(test_X)[:, 1]
    n_test = len(test_scores)
    test_pred_ranges = np.zeros((n_test, 2))
    scores_in_range = np.zeros(n_test)

    venn_info = model._venn_info
    calib_y = venn_info[:, 0]
    calib_scores = venn_info[:, 1]
    # TODO: check what weights I'm using here!
    calib_weights = venn_info[:, 2]

    # get the set of unique scores in the calibration set
    calib_scores_sorted = np.sort(list(set(calib_scores)))
    n_calib_sorted = len(calib_scores_sorted)

    # create a set of exemplar test points that fall between the calibration points
    exemplars = []
    exemplars.append(calib_scores_sorted[0]-1)
    for i in range(n_calib_sorted-1):
        exemplars.append((calib_scores_sorted[i] + calib_scores_sorted[i+1])/2.0)
    exemplars.append(calib_scores_sorted[-1]+1)
    n_exemplars = len(exemplars)

    calib_sorted_ranges = np.zeros((n_calib_sorted, 2))
    exemplar_ranges = np.zeros((n_exemplars, 2))

    # get the IR outputs if we repeat a calibration point
    for i in range(n_calib_sorted):
        for proposed_label in [0, 1]:
            all_scores = np.r_[calib_scores, calib_scores_sorted[i]]
            all_labels = np.r_[calib_y, proposed_label]
            # only give the new item a weight of 1 (equivalent to a single annotation of 0 or 1)
            all_weights = np.r_[calib_weights, 1.0]

            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            calib_sorted_ranges[i, proposed_label] = ir.predict([all_scores[-1]])

    # get the IR outputs for points in between
    for i in range(n_exemplars):
        for proposed_label in [0, 1]:

            all_scores = np.r_[calib_scores, exemplars[i]]
            all_labels = np.r_[calib_y, proposed_label]
            # only give the new item a weight of 1 (equivalent to a single annotation of 0 or 1)
            all_weights = np.r_[calib_weights, 1.0]

            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            exemplar_ranges[i, proposed_label] = ir.predict([all_scores[-1]])

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(calib_scores, calib_y, c='k')
        for i in range(n_calib_sorted):
            ax.plot((calib_scores_sorted[i], calib_scores_sorted[i]), (calib_sorted_ranges[i, 0], calib_sorted_ranges[i, 1]), c='b')
        for i in range(1, n_exemplars-1):
            ax.plot((exemplars[i], exemplars[i]), (exemplar_ranges[i, 0], exemplar_ranges[i, 1]), c='g')

        if test_pred_ranges[i, 0] <= test_scores[i] <= test_pred_ranges[i, 1]:
            scores_in_range[i] = 1.0
        plt.show()

    # check which interval each test point falls into
    for i in range(n_test):
        index = bisect.bisect_left(calib_scores_sorted, test_scores[i])
        # if it is beyond the end of the test point, use the last exemplar
        if index == n_calib_sorted:
            test_pred_ranges[i, :] = exemplar_ranges[index, :]
        # otherwise, if the score is equal to a calibration score, use that
        elif test_scores[i] == calib_scores_sorted[index]:
            test_pred_ranges[i, :] = calib_sorted_ranges[index, :]
        # otherwise, use the corresponding exemplar
        else:
            test_pred_ranges[i, :] = exemplar_ranges[index, :]

    return test_pred_ranges


def estimate_probs_from_labels(project_dir, full_model, model_name, labels_df, calib_subset, test_subset, weights_df=None, calib_items=None, test_items=None, verbose=False, plot=False):
    # do IVAP by making new predictions on calibration data

    n_items, n_classes = labels_df.shape

    if calib_items is not None:
        labels_df = labels_df.loc[calib_items]
        n_items, n_classes = labels_df.shape

    # normalize labels to just count one each
    labels = labels_df.values.copy()
    labels = labels / np.reshape(labels.sum(axis=1), (n_items, 1))

    # weight all examples evenly, assuming that I'm using my version of IR
    if weights_df is not None and calib_items is not None:
        calib_weights = np.array(weights_df.loc[calib_items].values).reshape((n_items,))
    else:
        calib_weights = np.ones(n_items)

    model_dir = os.path.join(dirs.dir_models(project_dir), model_name)
    feature_signatures = fh.read_json(os.path.join(model_dir, 'features.json'))
    calib_features_dir = dirs.dir_features(project_dir, calib_subset)

    printv("Loading features", verbose)
    calib_feature_list = []
    for sig in feature_signatures:
        feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
        printv("Loading %s" % feature_def, verbose)
        name = feature_def.name
        calib_feature = features.load_from_file(input_dir=calib_features_dir, basename=name)
        printv("Initial shape = (%d, %d)" % calib_feature.get_shape(), verbose)

        # use only a subset of the items, if given
        if calib_items is not None:
            all_test_items = calib_feature.get_items()
            n_items = len(all_test_items)
            item_index = dict(zip(all_test_items, range(n_items)))
            indices_to_use = [item_index[i] for i in calib_items]
            printv("Taking subset of items", verbose)
            calib_feature = features.create_from_feature(calib_feature, indices_to_use)
            printv("New shape = (%d, %d)" % calib_feature.get_shape(), verbose)
        printv("Setting vocabulary", verbose)
        calib_feature.set_terms(sig['terms'])
        idf = None
        if feature_def.transform == 'tfidf':
            idf = sig['idf']
        word_vectors_prefix = sig['word_vectors_prefix']
        calib_feature.transform(feature_def.transform, idf=idf, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
        printv("Final shape = (%d, %d)" % calib_feature.get_shape(), verbose)
        calib_feature_list.append(calib_feature)

    features_concat = features.concatenate(calib_feature_list)
    if features_concat.sparse:
        calib_X = features_concat.get_counts().tocsr()
    else:
        calib_X = features_concat.get_counts()

    calib_y = labels[:, 1]

    printv("Feature matrix shape: (%d, %d)" % calib_X.shape, verbose)

    test_features_dir = dirs.dir_features(project_dir, test_subset)

    printv("Loading features", verbose)
    test_feature_list = []
    for sig in feature_signatures:
        feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
        printv("Loading %s" % feature_def, verbose)
        name = feature_def.name
        test_feature = features.load_from_file(input_dir=test_features_dir, basename=name)
        printv("Initial shape = (%d, %d)" % test_feature.get_shape(), verbose)

        # use only a subset of the items, if given
        if test_items is not None:
            all_test_items = test_feature.get_items()
            n_items = len(all_test_items)
            item_index = dict(zip(all_test_items, range(n_items)))
            indices_to_use = [item_index[i] for i in test_items]
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
        test_feature_list.append(test_feature)

    features_concat = features.concatenate(test_feature_list)
    if features_concat.sparse:
        test_X = features_concat.get_counts().tocsr()
    else:
        test_X = features_concat.get_counts()

    n_test, _ = test_X.shape
    printv("Feature matrix shape: (%d, %d)" % test_X.shape, verbose)

    if full_model._model_type == 'ensemble':
        props_in_range = []
        n_items, _ = test_X.shape
        n_models = full_model.get_n_models()
        test_pred_ranges_all = np.zeros([n_models, n_items, 2])
        for model_i, model_name in enumerate(full_model._models.keys()):

            current_model = full_model._models[model_name]
            calib_pred_probs = current_model.predict_probs(calib_X)
            n_calib, n_classes = calib_pred_probs.shape
            assert n_classes == 2
            calib_scores = calib_pred_probs[:, 1]

            test_pred_ranges_all[model_i, :, :] = get_pred_for_one_model(current_model, test_X, calib_y, calib_scores, calib_weights, plot=plot)

        geometric_means = np.zeros([n_items, 2])
        for i in range(n_items):
            geometric_means[i, 0] = np.exp(np.sum(np.log(1.0 - test_pred_ranges_all[:, i, 0])) / float(n_models))
            geometric_means[i, 1] = np.exp(np.sum(np.log(test_pred_ranges_all[:, i, 1])) / float(n_models))

        test_pred_ranges = np.zeros([n_items, 2])
        #test_pred_ranges[:, 1] = np.max(test_pred_ranges_all[:, :, 1], axis=0)
        #test_pred_ranges[:, 0] = np.min(test_pred_ranges_all[:, :, 0], axis=0)
        #test_pred_ranges[:, 0] = 1.0 - geometric_means[:, 0]
        #test_pred_ranges[:, 1] = geometric_means[:, 1]
        test_pred_ranges[:, 1] = np.mean(test_pred_ranges_all[:, :, 1], axis=0)
        test_pred_ranges[:, 0] = np.mean(test_pred_ranges_all[:, :, 0], axis=0)

        combo = geometric_means[:, 1] / (geometric_means[:, 0] + geometric_means[:, 1])

    else:
        calib_pred_probs = full_model.predict_probs(calib_X)
        n_calib, n_classes = calib_pred_probs.shape
        assert n_classes == 2
        calib_scores = calib_pred_probs[:, 1]

        test_pred_ranges = get_pred_for_one_model(full_model, test_X, calib_y, calib_scores, calib_weights, plot=plot)
        combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])

    return test_pred_ranges, combo


def get_pred_for_one_model(model, test_X, calib_y, calib_scores, calib_weights, plot=False):
    test_scores = model.predict_probs(test_X)[:, 1]
    n_test = len(test_scores)
    test_pred_ranges = np.zeros((n_test, 2))
    scores_in_range = np.zeros(n_test)

    # get the set of unique scores in the calibration set
    calib_scores_sorted = np.sort(list(set(calib_scores)))
    n_calib_sorted = len(calib_scores_sorted)

    # create a set of exemplar test points that fall between the calibration points
    exemplars = []
    exemplars.append(calib_scores_sorted[0]-1)
    for i in range(n_calib_sorted-1):
        exemplars.append((calib_scores_sorted[i] + calib_scores_sorted[i+1])/2.0)
    exemplars.append(calib_scores_sorted[-1]+1)
    n_exemplars = len(exemplars)

    calib_sorted_ranges = np.zeros((n_calib_sorted, 2))
    exemplar_ranges = np.zeros((n_exemplars, 2))

    # get the IR outputs if we repeat a calibration point
    for i in range(n_calib_sorted):
        for proposed_label in [0, 1]:
            all_scores = np.r_[calib_scores, calib_scores_sorted[i]]
            all_labels = np.r_[calib_y, proposed_label]
            # only give the new item a weight of 1 (equivalent to a single annotation of 0 or 1)
            all_weights = np.r_[calib_weights, 1.0]

            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            calib_sorted_ranges[i, proposed_label] = ir.predict([all_scores[-1]])

    # get the IR outputs for points in between
    for i in range(n_exemplars):
        for proposed_label in [0, 1]:

            all_scores = np.r_[calib_scores, exemplars[i]]
            all_labels = np.r_[calib_y, proposed_label]
            # only give the new item a weight of 1 (equivalent to a single annotation of 0 or 1)
            all_weights = np.r_[calib_weights, 1.0]

            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            exemplar_ranges[i, proposed_label] = ir.predict([all_scores[-1]])

    if plot:
        fig, ax = plt.subplots()
        ax.scatter(calib_scores, calib_y, c='k')
        for i in range(n_calib_sorted):
            ax.plot((calib_scores_sorted[i], calib_scores_sorted[i]), (calib_sorted_ranges[i, 0], calib_sorted_ranges[i, 1]), c='b')
        for i in range(1, n_exemplars-1):
            ax.plot((exemplars[i], exemplars[i]), (exemplar_ranges[i, 0], exemplar_ranges[i, 1]), c='g')

        if test_pred_ranges[i, 0] <= test_scores[i] <= test_pred_ranges[i, 1]:
            scores_in_range[i] = 1.0
        plt.show()

    # check which interval each test point falls into
    for i in range(n_test):
        index = bisect.bisect_left(calib_scores_sorted, test_scores[i])
        # if it is beyond the end of the test point, use the last exemplar
        if index == n_calib_sorted:
            test_pred_ranges[i, :] = exemplar_ranges[index, :]
        # otherwise, if the score is equal to a calibration score, use that
        elif test_scores[i] == calib_scores_sorted[index]:
            test_pred_ranges[i, :] = calib_sorted_ranges[index, :]
        # otherwise, use the corresponding exemplar
        else:
            test_pred_ranges[i, :] = exemplar_ranges[index, :]

    return test_pred_ranges



def estimate_probs_from_labels_cv(project_dir, full_model, model_name, labels_df, calib_subset, weights_df=None, calib_items=None, verbose=False, plot=False):
    # do IVAP by making new predictions on calibration data

    n_items, n_classes = labels_df.shape

    if calib_items is not None:
        labels_df = labels_df.loc[calib_items]
        n_items, n_classes = labels_df.shape

    # normalize labels to just count one each
    labels = labels_df.values.copy()
    labels = labels / np.reshape(labels.sum(axis=1), (n_items, 1))

    # weight all examples evenly, assuming that I'm using my version of IR
    if weights_df is not None and calib_items is not None:
        calib_weights = np.array(weights_df.loc[calib_items].values).reshape((n_items,))
    else:
        calib_weights = np.ones(n_items)

    model_dir = os.path.join(dirs.dir_models(project_dir), model_name)
    feature_signatures = fh.read_json(os.path.join(model_dir, 'features.json'))
    calib_features_dir = dirs.dir_features(project_dir, calib_subset)

    printv("Loading features", verbose)
    calib_feature_list = []
    for sig in feature_signatures:
        feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
        printv("Loading %s" % feature_def, verbose)
        name = feature_def.name
        calib_feature = features.load_from_file(input_dir=calib_features_dir, basename=name)
        printv("Initial shape = (%d, %d)" % calib_feature.get_shape(), verbose)

        # use only a subset of the items, if given
        if calib_items is not None:
            all_test_items = calib_feature.get_items()
            n_items = len(all_test_items)
            item_index = dict(zip(all_test_items, range(n_items)))
            indices_to_use = [item_index[i] for i in calib_items]
            printv("Taking subset of items", verbose)
            calib_feature = features.create_from_feature(calib_feature, indices_to_use)
            printv("New shape = (%d, %d)" % calib_feature.get_shape(), verbose)
        printv("Setting vocabulary", verbose)
        calib_feature.set_terms(sig['terms'])
        idf = None
        if feature_def.transform == 'tfidf':
            idf = sig['idf']
        word_vectors_prefix = sig['word_vectors_prefix']
        calib_feature.transform(feature_def.transform, idf=idf, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
        printv("Final shape = (%d, %d)" % calib_feature.get_shape(), verbose)
        calib_feature_list.append(calib_feature)

    features_concat = features.concatenate(calib_feature_list)
    if features_concat.sparse:
        calib_X = features_concat.get_counts().tocsr()
    else:
        calib_X = features_concat.get_counts()

    calib_y = labels[:, 1]

    printv("Feature matrix shape: (%d, %d)" % calib_X.shape, verbose)

    #calib_pred_probs = full_model.predict_probs(calib_X)
    #n_calib, n_classes = calib_pred_probs.shape
    #assert n_classes == 2

    #calib_scores = calib_pred_probs[:, 1]

    list_of_n_levels = []
    if full_model._model_type == 'ensemble':
        props_in_range = []

        n_items, _ = calib_X.shape
        n_models = full_model.get_n_models()
        test_pred_ranges_all = np.zeros([n_models, n_items, 2])
        for model_i, model_name in enumerate(full_model._models.keys()):

            current_model = full_model._models[model_name]
            calib_pred_probs = current_model.predict_probs(calib_X)
            n_calib, n_classes = calib_pred_probs.shape
            assert n_classes == 2
            calib_scores = calib_pred_probs[:, 1]

            # check how many of the scores are in the resulting ranges
            test_pred_ranges_all[model_i, :, :] = get_pred_for_one_model_cv(calib_y, calib_scores, calib_weights, plot=plot)
            scores_in_range = (calib_scores > test_pred_ranges_all[model_i, :, 0]) * (calib_scores < test_pred_ranges_all[model_i, :, 1])
            props_in_range.append(str(np.mean(scores_in_range)))

            # fit a quick isotonic regression to the calibration data here, to see how many levels there are:
            ir = IsotonicRegression(0, 1)
            ir.fit(calib_scores, calib_y, calib_weights)
            calib_pred = ir.predict(calib_scores)
            n_levels = len(set(calib_pred))
            list_of_n_levels.append(str(n_levels))

        geometric_means = np.zeros([n_items, 2])
        for i in range(n_items):
            geometric_means[i, 0] = np.exp(np.sum(np.log(1.0 - test_pred_ranges_all[:, i, 0])) / float(n_models))
            geometric_means[i, 1] = np.exp(np.sum(np.log(test_pred_ranges_all[:, i, 1])) / float(n_models))

        test_pred_ranges = np.zeros([n_items, 2])
        #test_pred_ranges[:, 1] = np.max(test_pred_ranges_all[:, :, 1], axis=0)
        #test_pred_ranges[:, 0] = np.min(test_pred_ranges_all[:, :, 0], axis=0)
        #test_pred_ranges[:, 0] = 1.0 - geometric_means[:, 0]
        #test_pred_ranges[:, 1] = geometric_means[:, 1]
        test_pred_ranges[:, 1] = np.mean(test_pred_ranges_all[:, :, 1], axis=0)
        test_pred_ranges[:, 0] = np.mean(test_pred_ranges_all[:, :, 0], axis=0)

        combo = geometric_means[:, 1] / (geometric_means[:, 0] + geometric_means[:, 1])

    else:
        calib_pred_probs = full_model.predict_probs(calib_X)
        n_calib, n_classes = calib_pred_probs.shape
        assert n_classes == 2
        calib_scores = calib_pred_probs[:, 1]

        test_pred_ranges = get_pred_for_one_model_cv(calib_y, calib_scores, calib_weights, plot=plot)
        combo = test_pred_ranges[:, 1] / (1.0 - test_pred_ranges[:, 0] + test_pred_ranges[:, 1])

        scores_in_range = (calib_scores > test_pred_ranges[:, 0]) * (calib_scores < test_pred_ranges[:, 1])
        props_in_range = [str(float(np.mean(scores_in_range)))]

        # fit a quick isotonic regression to the calibration data here, to see how many levels there are:
        ir = IsotonicRegression(0, 1)
        ir.fit(calib_scores, calib_y, calib_weights)
        calib_pred = ir.predict(calib_scores)
        n_levels = len(set(calib_pred))
        list_of_n_levels.append(str(n_levels))

    return test_pred_ranges, combo, props_in_range, list_of_n_levels


def get_pred_for_one_model_cv(calib_y, calib_scores, calib_weights, plot=0):
    n_calib = len(calib_y)
    test_pred_ranges = np.zeros((n_calib, 2))

    for i in range(n_calib):

        scores_copy = calib_scores.tolist()
        y_copy = calib_y.tolist()
        weights_copy = calib_weights.tolist()

        test_score = scores_copy.pop(i)
        y_copy.pop(i)
        weights_copy.pop(i)

        if plot > 1:
            fig, ax = plt.subplots()
        for proposed_label in [0, 1]:

            all_scores = np.r_[scores_copy, test_score]
            all_labels = np.r_[y_copy, proposed_label]
            # only give the new item a weight of 1 (equivalent to a single annotation of 0 or 1)
            all_weights = np.r_[weights_copy, 1.0]

            #slopes = isotonic_regression.isotonic_regression(all_scores, all_labels)
            #test_pred_ranges[i, proposed_label] = slopes[-1]

            # upweight duplicate scores to force scikit learn's IR to do the right thing
            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            test_pred_ranges[i, proposed_label] = ir.predict([all_scores[-1]])

            if plot > 1:
                x_vals = all_scores.copy().tolist()
                x_vals.sort()
                pred_vals = ir.predict(x_vals)
                if proposed_label == 0:
                    ax.scatter(all_scores[:-1], all_labels[:-1], s=7, alpha=0.6)
                    ax.plot(x_vals, np.mean(all_labels[:-1]) * np.ones_like(x_vals), 'k--')
                ax.scatter(all_scores[-1], all_labels[-1], s=7, alpha=0.6)
                ax.plot(x_vals, pred_vals)
        if plot > 1:
            ax.scatter([all_scores[-1], all_scores[-1]], test_pred_ranges[i, :], s=8)
            ax.plot([all_scores[-1], all_scores[-1]], test_pred_ranges[i, :], 'r')
            plt.show()
            time.sleep(2)

    if plot > 0:
        order = np.argsort(calib_scores)
        for i in range(n_calib):
            j = order[i]
            if test_pred_ranges[j, 0] < calib_scores[j] < test_pred_ranges[j, 1]:
                plt.plot([i, i], [test_pred_ranges[j, 0], test_pred_ranges[j, 1]], c='g')
            else:
                plt.plot([i, i], [test_pred_ranges[j, 0], test_pred_ranges[j, 1]], c='r')
            plt.scatter(i, calib_scores[j], c='k', alpha=0.5)
        plt.show()

    return test_pred_ranges

if __name__ == '__main__':
    main()
