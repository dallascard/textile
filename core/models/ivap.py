import os
import time
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


def estimate_probs_brute_force(project_dir, model, model_name, calib_subset, test_subset, label_name, calib_items=None, test_items=None, verbose=False):

    label_dir = dirs.dir_labels(project_dir, calib_subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label_name + '.csv'), index_col=0, header=0)
    estimate_probs_from_labels(project_dir, model, model_name, calib_subset, test_subset, labels_df, calib_items, test_items, verbose)


def estimate_probs_from_labels(project_dir, model, model_name, calib_subset, test_subset, labels_df, calib_items=None, test_items=None, verbose=False, weights_df=None, plot=False):

    n_items, n_classes = labels_df.shape

    if calib_items is not None:
        labels_df = labels_df.loc[calib_items]
        n_items, n_classes = labels_df.shape

    # normalize labels to just count one each
    labels = labels_df.values.copy()

    # weight examples by the total number of votes they have received
    calib_weights = labels.sum(axis=1).copy()

    labels = labels / np.reshape(labels.sum(axis=1), (n_items, 1))

    #if weights_df is not None and calib_items is not None:
    #    calib_weights = np.array(weights_df.loc[calib_items].values).reshape((n_items,))
    #else:
    #    calib_weights = np.ones(n_items)

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

    calib_pred_probs = model.predict_probs(calib_X)
    n_calib, n_classes = calib_pred_probs.shape
    assert n_classes == 2

    calib_scores = calib_pred_probs[:, 1]

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
    printv("Feature matrix shape: (%d, %d)" % calib_X.shape, verbose)

    test_pred_probs = model.predict_probs(test_X)[:, 1]

    n_test = len(test_pred_probs)
    test_pred_ranges = np.zeros((n_test, 2))
    #test_pred_ranges2 = np.zeros((n_test, 2))

    for i in range(n_test):
        if plot:
            fig, ax = plt.subplots()
        for proposed_label in [0, 1]:

            all_scores = np.r_[calib_scores, test_pred_probs[i]]
            all_labels = np.r_[calib_y, proposed_label]
            # only give the new item a weight of 1 (equivalent to a single annotation of 0 or 1)
            all_weights = np.r_[calib_weights, 1.0]

            #slopes = isotonic_regression.isotonic_regression(all_scores, all_labels)
            #test_pred_ranges[i, proposed_label] = slopes[-1]

            # upweight duplicate scores to force scikit learn's IR to do the right thing
            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            test_pred_ranges[i, proposed_label] = ir.predict([all_scores[-1]])

            if plot:
                x_vals = all_scores.copy().tolist()
                x_vals.sort()
                pred_vals = ir.predict(x_vals)
                if proposed_label == 0:
                    ax.scatter(all_scores[:-1], all_labels[:-1], s=7, alpha=0.6)
                    ax.plot(x_vals, np.mean(all_labels[:-1]) * np.ones_like(x_vals), 'k--')
                ax.scatter(all_scores[-1], all_labels[-1], s=7, alpha=0.6)
                ax.plot(x_vals, pred_vals)
        if plot:
            ax.scatter([all_scores[-1], all_scores[-1]], test_pred_ranges[i, :], s=8)
            ax.plot([all_scores[-1], all_scores[-1]], test_pred_ranges[i, :], 'r')
            plt.show()
            time.sleep(2)

    # do the same internally for the calibration data
    calib_pred_ranges = np.zeros((n_calib, 2))
    for i in range(n_calib):
        for proposed_label in [0, 1]:

            all_scores = np.r_[calib_scores]
            all_labels = np.r_[calib_y]
            # consider changing the one label
            all_labels[i] = proposed_label
            all_weights = np.r_[calib_weights]

            # upweight duplicate scores to force scikit learn's IR to do the right thing
            ir = IsotonicRegression(0, 1)
            ir.fit(all_scores, all_labels, all_weights)
            calib_pred_ranges[i, proposed_label] = ir.predict([all_scores[i]])

    #print(np.sum(test_pred_ranges != test_pred_ranges2), np.size(test_pred_ranges))
    #print(np.max(np.abs(test_pred_ranges - test_pred_ranges2)))
    #print(np.mean(np.abs(test_pred_ranges - test_pred_ranges2)))

    return test_pred_ranges, calib_pred_ranges


if __name__ == '__main__':
    main()
