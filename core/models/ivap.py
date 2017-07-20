from optparse import OptionParser

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    n = 30
    x = np.random.randint(low=0, high=11, size=n)
    #y = np.random.randint(low=0, high=2, size=n)
    y = np.zeros_like(x)
    for i in range(n):
        y[i] = np.random.binomial(n=1, p=expit(x[i] - 5.0))

    isotonic_regression(x, y)


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

            slopes, order = isotonic_regression(scores, labels)
            pred_ranges[i, proposed_label] = slopes[order[-1]]

    return pred_ranges


def isotonic_regression(scores, labels):

    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]
    n_scores = len(scores)

    weights, y_prime, csd_points = compute_csd(scores, labels)
    n_points, _ = csd_points.shape

    gcm, slopes = compute_gcm(csd_points)

    fig, ax = plt.subplots()
    ax.scatter(scores+np.random.randn(n_scores)*0.1, labels, s=5, edgecolor='k', facecolor='k', alpha=0.8)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(csd_points[:, 0], csd_points[:, 1])
    plt.plot(gcm[:, 0], gcm[:, 1])
    indices = gcm[:, 1] > csd_points[:, 1]
    if np.sum(indices) > 0:
        ax.scatter(csd_points[indices, 0], csd_points[indices, 1], c='r')
    #ax.set_xlim(0, n_points+1)
    #ax.set_ylim(0, n_points+1)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(slopes[1:])
    ax.set_ylim(-0.1, 1.1)
    plt.show()

    return slopes[1:], order


def compute_csd(scores, labels):
    score_set = list(set(scores))
    score_set.sort()

    n_scores = len(score_set)
    weights = np.zeros(n_scores+1)
    y_prime = np.zeros(n_scores+1)
    points = np.zeros([n_scores+1, 2])

    for s_i, s in enumerate(score_set):
        indices = scores == s
        weights[s_i+1] = np.sum(indices)
        y_prime[s_i+1] = np.sum(labels[indices]) / float(weights[s_i+1])
        points[s_i+1, 0] = np.sum(weights[:s_i+2])
        points[s_i+1, 1] = np.dot(weights[:s_i+2], y_prime[:s_i+2])

    return weights, y_prime, points


def compute_gcm(points):
    n_points, _ = points.shape
    min_point_index = np.argmin(points[:, 1])
    print("left")
    left_gcm = compute_left_gcm(points, min_point_index)
    print("right")
    right_gcm = compute_right_gcm(points, min_point_index)
    gcm_corners = left_gcm[:-1] + right_gcm

    gcm = np.zeros_like(points)
    slopes = np.zeros(n_points)
    for i, left_index in enumerate(gcm_corners[:-1]):
        right_index = gcm_corners[i+1]
        slope = (points[left_index, 1] - points[right_index, 1]) / float(points[left_index, 0] - points[right_index, 0])
        for j in range(left_index, right_index+1):
            gcm[j, 0] = points[j, 0]
            gcm[j, 1] = points[left_index, 1] + (points[j, 0] - points[left_index, 0]) * slope
            slopes[j] = slope
    #gcm[-1, :] = points[-1, :]
    return gcm, slopes


def compute_left_gcm(points, index):
    """Compute the GCM for points to the left of index"""
    if index == 0:
        return [index]
    else:
        n_points, _ = points.shape
        slopes = np.zeros(n_points)
        for i in range(0, index):
            slopes[i] = (points[index, 1] - points[i, 1]) / float(points[index, 0] - points[i, 0])
        print(slopes)
        max_slope_index = np.argmax(slopes[:index])
        print(max_slope_index)
        return compute_left_gcm(points, max_slope_index) + [index]


def compute_right_gcm(points, index):
    """Compute the GCM for points to the right of index"""
    n_points, _ = points.shape
    if index == n_points - 1:
        return [index]
    else:
        slopes = np.zeros(n_points)
        for i in range(index+1, n_points):
            slopes[i] = (points[i, 1] - points[index, 1]) / float(points[i, 0] - points[index, 0])
        print(slopes)
        min_slope_index = np.argmin(slopes[index+1:]) + index + 1
        print(min_slope_index)
        return [index] + compute_right_gcm(points, min_slope_index)




if __name__ == '__main__':
    main()
