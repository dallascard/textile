from optparse import OptionParser

import numpy as np

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

    for i in range(1):
        scores = np.r_[calib_pred_probs, test_pred_probs[i]]
        labels_0 = np.r_[calib_y, 0]
        labels_1 = np.r_[calib_y, 1]

        order = np.argsort(scores)
        test_pos = order[-1]

        scores = scores[order]
        labels_0 = labels_0[order]
        labels_1 = labels_1[order]

        score_set = list(set(scores))
        score_set.sort()

        n_scores = len(score_set)
        weights = np.zeros(n_scores)
        y_prime_0 = np.zeros(n_scores)
        y_prime_1 = np.zeros(n_scores)
        points_0 = np.zeros(n_scores, 2)
        points_1 = np.zeros(n_scores, 2)

        for s_i, s in enumerate(score_set):
            indices = scores == s
            weights[s_i] = np.sum(indices)
            y_prime_0[s_i] = np.sum(labels_0[indices])
            y_prime_1[s_i] = np.sum(labels_1[indices])
            points_0[s_i, 0] = np.sum(weights[:s_i+1])
            points_0[s_i, 1] = np.dot(weights[:s_i+1], y_prime_0[:s_i+1])
            points_1[s_i, 0] = np.sum(weights[:s_i+1])
            points_1[s_i, 1] = np.dot(weights[:s_i+1], y_prime_1[:s_i+1])




if __name__ == '__main__':
    main()
