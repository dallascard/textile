import os
import sys
from optparse import OptionParser

import numpy as np

from ..util import file_handling as fh
from ..preprocessing import features
from ..main import train, predict, evaluate_predictions, estimate_proportions
from ..util import dirs


def main():
    usage = "%prog project_dir subset field_name model_name config.json "
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='calib_prop', default=0.5,
                      help='Percent to use for the calibration part of each split: default=%default')
    parser.add_option('--sampling', dest='sampling', default='proportional',
                      help='How to divide calibration and test data [proportional|random]: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|BLR]: default=%default')
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='Covariate shift method [None]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
                      help='Use to fit a model with no intercept: default=%default')
    #parser.add_option('--objective', dest='objective', default='f1',
    #                  help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--n_classes', dest='n_classes', default=None,
                      help='Specify the number of classes (None=max[training labels]): default=%default')
    parser.add_option('--n_dev_folds', dest='n_dev_folds', default=5,
                      help='Number of dev folds for tuning regularization: default=%default')
    parser.add_option('--seed', dest='seed', default=None,
                      help='Random seed (None=random): default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    field_name = args[2]
    model_name = args[3]
    config_file = args[4]

    calib_percent = float(options.calib_prop)
    sampling = options.sampling
    model_type = options.model
    label = options.label
    penalty = options.penalty
    cshift = options.cshift
    #objective = options.objective
    intercept = not options.no_intercept
    n_classes = options.n_classes
    if n_classes is not None:
        n_classes = int(n_classes)
    n_dev_folds = int(options.n_dev_folds)
    if options.seed is not None:
        np.random.seed(int(options.seed))

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    weights_file = None

    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata[field_name].values))
    field_vals.sort()
    print(field_vals)
    subset_5 = metadata[metadata[field_name] == field_vals[5]]
    train_items = subset_5.index
    subset_6 = metadata[metadata[field_name] == field_vals[6]]
    test_items = subset_6.index

    print("Doing training")
    model = train.train_model(project_dir, model_type, model_name, subset, label, feature_defs, weights_file, items_to_use=train_items, n_classes=n_classes, penalty=penalty, intercept=intercept, n_dev_folds=n_dev_folds)

    print("Doing evaluation")
    predictions, pred_probs = predict.predict(project_dir, model, model_name, subset, label, items_to_use=test_items)

    evaluate_predictions.load_and_evaluate_predictons(project_dir, model_name, subset, label, items_to_use=test_items, n_classes=n_classes, pos_label=1, average='micro')


if __name__ == '__main__':
    main()
