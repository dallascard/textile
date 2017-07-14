import sys
import numpy as np

from sklearn.metrics import f1_score as skl_f1_score
from sklearn.metrics import accuracy_score as skl_acc_score


def f1_score(true, pred, n_classes=2, pos_label=1, average=None):
    """
    Override f1_score in sklearn in order to deal with both binary and multiclass cases
    :param true: true labels
    :param pred: predicted labels
    :param n_classes: total number of different possible labels
    :param pos_label: label to use as the positive label for the binary case (0 or 1)
    :param average: how to calculate f1 for the multiclass case (default = 'micro')

    :return: f1 score
    """

    if n_classes == 2:
        if np.sum(true * pred) == 0:
            #print("Warning: no true positives")
            f1 = 0.0
        else:
            f1 = skl_f1_score(true, pred, average='binary', labels=range(n_classes), pos_label=pos_label)
    else:
        if average is None:
            f1 = skl_f1_score(true, pred, average='micro', labels=range(n_classes), pos_label=None)
        else:
            f1 = skl_f1_score(true, pred, average=average, labels=range(n_classes), pos_label=None)
    return f1


def acc_score(true, pred, n_classes=2):
    acc = skl_acc_score(np.array(true, dtype=int), np.array(pred, dtype=int))
    return acc


def brier_score(true, pred_probs, binary_form=False):
    n_items, n_classes = pred_probs.shape
    true_probs = np.zeros_like(pred_probs)
    true_probs[range(n_items), true] = 1.0
    # this is the original definition given in Brier
    score = np.mean((true_probs - pred_probs)**2)
    if binary_form:
        # this is the more commonly used form in the binary case
        score /= score
    return score


def evaluate_calibration_mse_bins(true_labels, pred_probs, n_bins=5):
    n_items, n_classes = pred_probs.shape
    if n_items < n_bins:
        n_bins = n_items

    breakpoints = list(np.array(np.arange(n_bins)/float(n_bins) * n_items, dtype=int).tolist()) + [n_items]
    mse_sum = 0
    for label in range(n_classes):
        label_probs = pred_probs[:, label]
        order = np.argsort(label_probs)

        for b in range(n_bins):
            start = breakpoints[b]
            end = breakpoints[b+1]
            items = order[start:end]
            mean_bin_probs = np.mean(pred_probs[items, label])
            mean_bin_labels = np.mean(np.array(true_labels == label, dtype=float)[items])
            mse = (mean_bin_labels - mean_bin_probs)**2
            mse_sum += mse

    mse = np.sqrt(mse_sum/float(n_bins)/float(n_classes))

    return mse


def evaluate_calibration_mse(true_labels, pred_probs, min_bins=3, max_bins=5):
    return np.mean([evaluate_calibration_mse_bins(true_labels, pred_probs, n_bins) for n_bins in range(min_bins, max_bins+1)])


def mean_mae(true_props, pred_props):
    return np.mean(np.abs(np.array(true_props) - np.array(pred_props)))


def eval_proportions(true_probs, pred_probs, metric):
    if metric == 'kld':
        eval_func = eval_proportions_kld
    elif metric == 'mse':
        eval_func = eval_proportions_mse
    elif metric == 'mae':
        eval_func = eval_proportions_mae
    else:
        sys.exit("Evaluation metric not recognized")
    return eval_func(true_probs, pred_probs)


def eval_proportions_mse(true_props, pred_props):
    return np.sqrt(np.mean((np.array(true_props) - np.array(pred_props))**2))


def eval_proportions_mae(true_props, pred_props):
    return np.mean(np.abs(np.array(true_props) - np.array(pred_props)))


def eval_proportions_kld(true_props, pred_props, epsilon=1e-5):
    kld_sum = 0
    for i, p in enumerate(true_props):
        kld_sum += (p+epsilon) * (np.log(p+epsilon) - np.log(pred_props[i]+epsilon))
    return kld_sum


def compute_proportions(labels, n_classes):
    label_counts = np.bincount(labels.values.reshape(len(labels),), minlength=n_classes)
    proportions = label_counts / float(label_counts.sum())
    return proportions