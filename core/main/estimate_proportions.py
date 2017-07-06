import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import dirs
from ..util import file_handling as fh
from ..models import lr, blr, load_model
from ..preprocessing import features


def main():
    usage = "%prog project_dir predict_subset model_name"
    parser = OptionParser(usage=usage)
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|BLR]: default=%default')
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

    # TODO: make this automatic
    model_type = options.model
    label = options.label
    load_and_predict(project_dir, model_type, model_name, predict_subset, label)


def load_and_predict(project_dir, model_type, model_name, test_subset, label_name):
    print("Loading model")

    model_dir = os.path.join(dirs.dir_models(project_dir), model_name)
    model = load_model.load_model(model_dir, model_type)
    model_type = model.get_model_type()

    feature_signatures = fh.read_json(os.path.join(model_dir, 'features.json'))
    test_features_dir = dirs.dir_features(project_dir, test_subset)

    print("Loading features")
    feature_list = []
    for sig in feature_signatures:
        feature_def = features.FeatureDef(sig['name'], sig['min_df'], sig['max_fp'], sig['transform'])
        print("Loading %s" % feature_def)
        name = feature_def.name
        test_feature = features.load_from_file(input_dir=test_features_dir, basename=name)
        test_feature.set_terms(sig['terms'])
        idf = None
        if feature_def.transform == 'tfidf':
            idf = sig['idf']
        test_feature.transform(feature_def.transform, idf=idf)
        feature_list.append(test_feature)

    features_concat = features.concatenate(feature_list)
    X = features_concat.get_counts().tocsr()

    if model_type == 'BLR':
        X = np.array(X.todense())

    print("Doing prediction")
    if model_type == 'LR':
        pred_probs = model.predict_probs(X)
        n_items, n_classes = pred_probs.shape
        mean_probs = np.mean(pred_probs, axis=0)
        est_stdevs = np.zeros_like(mean_probs)
        for c in range(n_classes):
            est_stdevs[c] = np.sqrt(np.sum([p * (1-p) for p in pred_probs[c, :]])/float(n_items))
        print("Mean probs:", mean_probs)
        print("Est stdevs:", est_stdevs)
    elif model_type == 'BLR':
        pred_probs = model.sample_probs(X, n_samples=100)
        mean_probs = np.mean(pred_probs, axis=0)
        print("Mean probs:", np.mean(mean_probs))
        print("Est stdevs:", np.std(mean_probs))


if __name__ == '__main__':
    main()
