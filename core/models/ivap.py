from optparse import OptionParser

import numpy as np

from ..models import isotonic_regression


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()


def estimate_probs_brute_force(model, calib_X, calib_y, test_X):
    calib_pred_probs = model.predict_probs(calib_X)
    n_calib, n_classes = calib_pred_probs.shape
    assert n_classes == 2

    calib_pred_probs = calib_pred_probs[:, 1]

    test_pred_probs = model.predict_probs(test_X)[:, 1]
    n_test = len(test_pred_probs)
    pred_ranges = np.zeros(n_test, 2)

    for i in range(n_test):
        for proposed_label in [0, 1]:

            scores = np.r_[calib_pred_probs, test_pred_probs[i]]
            labels = np.r_[calib_y, proposed_label]

            order = np.argsort(scores)
            scores = scores[order]
            labels = labels[order]

            slopes = isotonic_regression.isotonic_regression(scores, labels)
            pred_ranges[i, proposed_label] = slopes[order[-1]]

    return pred_ranges


if __name__ == '__main__':
    main()
