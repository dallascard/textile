import os
import sys
from optparse import OptionParser
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import sparse

from ..util import dirs
from ..util import file_handling as fh
from ..models import load_model
from ..preprocessing import features
from ..util.misc import printv
from ..main import train


def main():
    usage = "%prog project_dir predict_subset model_name"
    parser = OptionParser(usage=usage)
    parser.add_option('--model', dest='model', default=None,
                      help='Model type [LR|BLR]; None=auto-detect: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    project_dir = args[0]
    predict_subset = args[1]
    model_name = args[2]

    model_type = options.model
    label = options.label

    load_and_predict(project_dir, model_type, model_name, predict_subset, label)


def load_and_predict(project_dir, model_type, model_name, test_subset, label_name, items_to_use=None):
    print("Loading model")
    model_dir = os.path.join(dirs.dir_models(project_dir), model_name)
    model = load_model.load_model(model_dir, model_name, model_type)
    predict(project_dir, model, model_name, test_subset, label_name, items_to_use=items_to_use)


def predict(project_dir, model, model_name, test_subset, label_name, items_to_use=None, verbose=False, force_dense=False, group_identical=False, n_samples=10):

    model_dir = os.path.join(dirs.dir_models(project_dir), model_name)

    feature_signatures = fh.read_json(os.path.join(model_dir, 'features.json'))
    test_features_dir = dirs.dir_features(project_dir, test_subset)

    printv("Loading features", verbose)
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

    features_concat = features.concatenate(feature_list)
    if features_concat.sparse:
        X = features_concat.get_counts().tocsr()
    else:
        X = features_concat.get_counts()
    if force_dense:
        assert X.size < 10000000
        X = np.array(X.todense())

    n_items, n_labels = X.shape
    print("Feature matrix shape: (%d, %d)" % X.shape)

    """
    if group_identical:
        label_dir = dirs.dir_labels(project_dir, test_subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label_name + '.csv'), index_col=0, header=0)
        Y = labels_df.loc[items_to_use].as_matrix()

        X_counts = defaultdict(int)
        X_n_pos = defaultdict(int)

        for i in range(n_items):
            if sparse.issparse(X):
                vector = np.array(X[i, :].todense()).ravel()
            else:
                vector = np.array(X[i, :]).ravel()
            key = ''.join([str(int(s)) for s in vector])
            X_counts[key] += np.sum(Y[i, :])
            X_n_pos[key] += Y[i, 1]

        keys = list(X_counts.keys())
        keys.sort()

        train_patterns = fh.read_json(os.path.join(model_dir, 'training_patterns.json'))
        X_counts_train = train_patterns['X_counts']
        X_n_pos_train = train_patterns['X_n_pos']

        patterns = list(X_counts.keys())
        counts = [X_counts[key] for key in patterns]
        order = list(np.argsort(counts))
        order.reverse()
        most_common_keys = [patterns[i] for i in order[:10]]
        for key in most_common_keys:
            print('key: %s' % key)
            print('test: %f (%d / %d)' % (X_n_pos[key] / float(X_counts[key]),  X_n_pos[key], X_counts[key]))
            if key in X_n_pos_train:
                print('train: %f (%d / %d)' % (X_n_pos_train[key] / float(X_counts_train[key]),  X_n_pos_train[key], X_counts_train[key]))
            else:
                print('train: %f (%d / %d)' % (0.0, 0, 0))
            prob = model.predict_probs(sparse.csr_matrix([int(i) for i in key]))
            print('pred:', prob)

        #for index, item in enumerate(items_to_use):
        #    print(X[index], item, Y[index])
    """

    if group_identical:
        print("One by one")
        label_dir = dirs.dir_labels(project_dir, test_subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label_name + '.csv'), index_col=0, header=0)
        Y = labels_df.loc[items_to_use].as_matrix()

        X_counts = defaultdict(int)
        X_n_pos = defaultdict(int)

        for i in range(n_items):
            if sparse.issparse(X):
                vector = np.array(X[i, :].todense()).ravel()
            else:
                vector = np.array(X[i, :]).ravel()
            key = ''.join([str(int(s)) for s in vector])
            X_counts[key] += np.sum(Y[i, :])
            X_n_pos[key] += Y[i, 1]

        keys = list(X_counts.keys())
        keys.sort()

        train_patterns = fh.read_json(os.path.join(model_dir, 'training_patterns.json'))
        X_counts_train = train_patterns['X_counts']
        X_n_pos_train = train_patterns['X_n_pos']

        patterns = list(X_counts.keys())
        counts = [X_counts[key] for key in patterns]
        order = list(np.argsort(counts))
        order.reverse()
        most_common_keys = [patterns[i] for i in order[:10]]
        for key in most_common_keys:
            print('key: %s' % key)
            print('test: %f (%d / %d)' % (X_n_pos[key] / float(X_counts[key]),  X_n_pos[key], X_counts[key]))
            if key in X_n_pos_train:
                print('train: %f (%d / %d)' % (X_n_pos_train[key] / float(X_counts_train[key]),  X_n_pos_train[key], X_counts_train[key]))
            else:
                print('train: %f (%d / %d)' % (0.0, 0, 0))
            prob = model.predict_probs(sparse.csr_matrix([int(i) for i in key]))
            print('pred:', prob)

    print("Doing prediction")
    predictions = model.predict(X)
    pred_probs = model.predict_probs(X)
    n_items, n_labels = pred_probs.shape

    output_dir = dirs.dir_predictions(project_dir, test_subset, model_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    predictions_df = pd.DataFrame(predictions, index=features_concat.get_items(), columns=[label_name])
    predictions_df.to_csv(os.path.join(output_dir, label_name + '_predictions.csv'))

    pred_probs_df = pd.DataFrame(pred_probs, index=features_concat.get_items(), columns=range(n_labels))
    pred_probs_df.to_csv(os.path.join(output_dir, label_name + '_pred_probs.csv'))
    print("Done")

    pred_proportions = model.predict_proportions(X)
    samples = None

    if model.get_model_type() == 'DL':
        print("Testing DL model")
        label_dir = dirs.dir_labels(project_dir, test_subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label_name + '.csv'), index_col=0, header=0)
        Y = labels_df.loc[items_to_use].as_matrix()
        n_items, n_classes = Y.shape
        weights = np.ones(n_items)
        X, Y, w = train.prepare_data(X, Y, weights=weights, loss='log', normalize=True)
        model.test(X, Y, w)

        samples = model.sample(X)

    elif model.get_model_type() == 'ensemble':
        models = model._models

        samples = []
        label_dir = dirs.dir_labels(project_dir, test_subset)
        labels_df = fh.read_csv_to_df(os.path.join(label_dir, label_name + '.csv'), index_col=0, header=0)
        Y = labels_df.loc[items_to_use].as_matrix()
        n_items, n_classes = Y.shape
        weights = np.ones(n_items)
        X_norm, Y_norm, w_norm = train.prepare_data(X, Y, weights=weights, loss='log', normalize=True)

        for name, m in models.items():
            if m.get_model_type() == 'DL':
                print(name)
                m.test(X_norm, Y_norm, w_norm)
                samples_m = m.sample(X)
                samples.append(samples_m)

        if len(samples) > 0:
            samples = np.hstack(samples)
        else:
            samples = None

    return predictions_df, pred_probs_df, pred_proportions, samples


if __name__ == '__main__':
    main()
