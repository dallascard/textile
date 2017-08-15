import os
from optparse import OptionParser

import numpy as np

from ..util import dirs
from ..util import file_handling as fh
from ..util.misc import printv
from ..models import evaluation
from ..main import train


def main():
    usage = "%prog project subset model_name"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--pos_label', dest='pos_label', default=1,
                      help='Positive label (binary only): default=%default')
    parser.add_option('--average', dest='average', default='micro',
                      help='Averageing to use for F1 (multiclass only): default=%default')


    (options, args) = parser.parse_args()
    project_dir = args[0]
    subset = args[1]
    model_name = args[2]
    label = options.label
    pos_label = int(options.pos_label)
    average = options.average

    load_and_evaluate_predictons(project_dir, model_name, subset, label, pos_label=pos_label, average=average)


def load_and_evaluate_predictons(project_dir, model_name, subset, label, items_to_use=None, pos_label=1, average='micro'):

    label_dir = dirs.dir_labels(project_dir, subset)
    labels = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    pred_dir = dirs.dir_predictions(project_dir, subset, model_name)
    predictions = fh.read_csv_to_df(os.path.join(pred_dir, label + '_predictions.csv'), index_col=0, header=0)
    pred_probs = fh.read_csv_to_df(os.path.join(pred_dir, label + '_pred_probs.csv'), index_col=0, header=0)

    if items_to_use is not None:
        labels = labels.loc[items_to_use]
        predictions = predictions.loc[items_to_use]
        pred_probs = pred_probs.loc[items_to_use]

    weights = None

    evaluate_predictions(labels, predictions, pred_probs_df=pred_probs, pos_label=pos_label, average=average, weights=weights)


def evaluate_predictions(labels_df, predictions_df, pred_probs_df=None, pos_label=1, average='micro', weights=None):
    assert np.all(labels_df.index == predictions_df.index)
    n_items, n_classes = labels_df.shape
    labels = labels_df.values
    predictions = predictions_df.values.reshape((n_items,))


    if pred_probs_df is None:
        pred_probs = np.zeros([n_items, 2])
    else:
        pred_probs = pred_probs_df.values

    pred_probs, labels, weights, predictions = train.expand_features_and_labels(pred_probs, labels, weights, predictions)
    n_items, _ = labels.shape
    true = np.argmax(labels, axis=1)

    f1 = evaluation.f1_score(true, predictions, n_classes, pos_label=pos_label, average=average, weights=weights)
    print("F1 = %0.3f" % f1)

    acc = evaluation.acc_score(true, predictions, n_classes, weights=weights)
    print("Accuracy = %0.3f" % acc)

    rmse = evaluation.evaluate_proportions_mse(true, predictions, n_classes, weights, verbose=True)
    print("RMSE on proportions = %0.3f" % rmse)

    if pred_probs is not None:
        if weights is None:
            weights = np.ones(n_items)
        pred_props = np.dot(weights, pred_probs)
        pred_props = pred_probs / np.sum(pred_props)
        true_props = evaluation.compute_proportions(labels, weights)
        rmse = evaluation.eval_proportions(true_props, pred_props, 'mse')
        print("RMSE on \sum pred probs = %0.3f" % rmse)


    return f1, acc


if __name__ == '__main__':
    main()
