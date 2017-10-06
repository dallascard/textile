import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import dirs
from ..util import file_handling as fh
from ..models import load_model
from ..preprocessing import features
from ..util.misc import printv


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


def predict(project_dir, model, model_name, test_subset, label_name, items_to_use=None, verbose=False, force_dense=False):

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
        X = np.array(X.todense())

    print("Feature matrix shape: (%d, %d)" % X.shape)

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

    return predictions_df, pred_probs_df, pred_proportions


if __name__ == '__main__':
    main()
