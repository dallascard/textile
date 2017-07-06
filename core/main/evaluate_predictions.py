import os
from optparse import OptionParser

import numpy as np

from ..util import dirs
from ..util import file_handling as fh

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

    label_dir = dirs.dir_labels(project_dir, subset)
    labels = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    pred_dir = dirs.dir_predictions(project_dir, subset, model_name)
    predictions = fh.read_csv_to_df(os.path.join(pred_dir, label + '_predictions.csv'), index_col=0, header=0)

    evaluate_predictions(labels, predictions, n_classes=n_classes, pos_label=pos_label, average=average)


def evaluate_predictions(labels, predictions, n_classes=None, pos_label=1, average='micro'):
    assert np.all(labels.index == predictions.index)
    if n_classes is None:
        n_classes = np.max([np.max(labels), np.max(predictions)]) + 1
        print("Assuming %d classes" % n_classes)
    f1 = evaluation.f1_score(labels, predictions, n_classes, pos_label=pos_label, average=average)
    print("F1 = %0.3f" % f1)
    acc = evaluation.acc_score(labels, predictions, n_classes)
    print("Accuracy = %0.3f" % acc)

    true_label_counts = np.bincount(labels, minlength=n_classes)
    true_proportions = true_label_counts / float(true_label_counts.sum())
    print("True proportions =", true_proportions)

    pred_label_counts = np.bincount(predictions, minlength=n_classes)
    pred_proportions = pred_label_counts / float(pred_label_counts.sum())
    print("Predicted proportions =", pred_proportions)

    mse = np.mean((pred_proportions - true_proportions) ** 2)
    print("MSE on proportions = %0.3f" % mse)


if __name__ == '__main__':
    main()
