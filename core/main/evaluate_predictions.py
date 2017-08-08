import os
from optparse import OptionParser

import numpy as np

from ..util import dirs
from ..util import file_handling as fh
from ..util.misc import printv
from ..models import evaluation


def main():
    usage = "%prog project subset model_name"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('-n', dest='n_classes', default=None,
                      help='Number of classes (None=max+1): default=%default')
    parser.add_option('--pos_label', dest='pos_label', default=1,
                      help='Positive label (binary only): default=%default')
    parser.add_option('--average', dest='average', default='micro',
                      help='Averageing to use for F1 (multiclass only): default=%default')


    (options, args) = parser.parse_args()
    project_dir = args[0]
    subset = args[1]
    model_name = args[2]
    label = options.label
    n_classes = options.n_classes
    if n_classes is not None:
        n_classes = int(n_classes)
    pos_label = int(options.pos_label)
    average = options.average

    load_and_evaluate_predictons(project_dir, model_name, subset, label, n_classes=n_classes, pos_label=pos_label, average=average)


def load_and_evaluate_predictons(project_dir, model_name, subset, label, items_to_use=None, n_classes=None, pos_label=1, average='micro'):

    label_dir = dirs.dir_labels(project_dir, subset)
    labels = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    pred_dir = dirs.dir_predictions(project_dir, subset, model_name)
    predictions = fh.read_csv_to_df(os.path.join(pred_dir, label + '_predictions.csv'), index_col=0, header=0)

    if items_to_use is not None:
        labels = labels.loc[items_to_use]
        predictions = predictions.loc[items_to_use]

    evaluate_predictions(labels, predictions, pos_label=pos_label, average=average)


def evaluate_predictions(labels_df, predictions_df, pos_label=1, average='micro', weights_df=None):
    assert np.all(labels_df.index == predictions_df.index)
    n_items, n_classes = labels_df.shape
    labels = labels_df.values
    predictions = predictions_df.values.reshape((n_items,))

    labels_per_item = labels.sum(axis=1)
    if weights_df is None:
        weights = np.array(1.0 / labels_per_item)
    else:
        weights = weights_df.values
        weights = weights * np.array(1.0 / labels_per_item)

    labels_list = []
    pred_list = []
    weights_list = []

    for c in range(n_classes):
        c_max = np.max(labels[:, c])
        for i in range(c_max):
            items = np.array(labels[:, c] > i)
            labels_list.append(np.ones(np.sum(items), dtype=int) * c)
            pred_list.append(predictions[items])
            weights_list.append(weights[items])

    #print(labels_list)
    #print(pred_list)
    #print(weights_list)
    labels = np.hstack(labels_list)
    predictions = np.hstack(pred_list)
    weights = np.hstack(weights_list)

    f1 = evaluation.f1_score(labels, predictions, n_classes, pos_label=pos_label, average=average, weights=weights)
    print("F1 = %0.3f" % f1)

    acc = evaluation.acc_score(labels, predictions, n_classes, weights=weights)
    print("Accuracy = %0.3f" % acc)

    rmse = evaluation.evaluate_proportions_mse(labels, predictions, n_classes, weights)
    print("RMSE on proportions = %0.3f" % rmse)

    return f1, acc


if __name__ == '__main__':
    main()
