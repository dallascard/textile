from optparse import OptionParser

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()


    # Some code to test isotonic regression
    n = 20
    # create some random Bernoulli data with repeated x values
    x = np.random.randint(low=0, high=11, size=n)
    #x = np.arange(n)
    y = np.zeros_like(x)
    for i in range(n):
        y[i] = np.random.binomial(n=1, p=expit((x[i] - 5.0)/2.0))

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    print(x)
    print(y)

    # compute the isotonic regression through these poitns
    isotonic_regression(x, y, plot=True)


def isotonic_regression(scores, labels, plot=False):
    """
    Compute the isotonic regression for a set of points (s, y)
    :param scores: a length-k vector of scores \in (-\inf, \inf) 
    :param labels: a length-k vector of corresponding labels \in {0, 1} (also works outside this range)
    :return: the value of the isotonic regression at each point
    """

    # compute the cumulative sum diagram (CSD)
    score_set, csd_points, weights = compute_csd(scores, labels)

    # compute the greatest convex minorant (GCM) and it's slope to the right of each point
    gcm, slopes = compute_gcm(csd_points)

    print(gcm)

    if plot:
        # plot the CSD and GCM
        fig, axes = plt.subplots(1, 2)
        ax1, ax2 = axes
        ax1.scatter(csd_points[:, 0], csd_points[:, 1])
        ax1.plot(gcm[:, 0], gcm[:, 1])

        # plot the original data with jitter
        jitter = np.random.randn(len(scores)) * (np.max(labels) - np.min(labels)) * 0.015
        ax2.scatter(scores, labels + jitter, s=8, edgecolor='k', facecolor='b', alpha=0.5)
        # overlay the value of the isotonic regression
        ax2.plot(score_set, slopes[1:])

        full_pred = []
        for i, s in enumerate(score_set):
            full_pred.extend([float(slopes[i+1])] * int(weights[i+1]))
        se = (np.array(full_pred) - labels)**2
        print(se)
        print(np.mean(se))

        ir = IsotonicRegression()
        ir.fit(scores, labels)
        y_pred = ir.predict(score_set)

        full_pred = []
        for i, s in enumerate(score_set):
            full_pred.extend([float(y_pred[i])] * int(weights[i+1]))
        se = (np.array(full_pred) - labels)**2
        print(se)
        print(np.mean(se))

        ax2.plot(score_set, y_pred, 'k--')

        # plot the CSD and GCM
        gcm_sklearn = csd_points.copy()
        for i in range(1, len(score_set)+1):
            gcm_sklearn[i, 1] = gcm_sklearn[i-1, 1] + y_pred[i-1] * (csd_points[i, 0] - csd_points[i-1, 0])
        ax1.plot(gcm_sklearn[:, 0], gcm_sklearn[:, 1], 'k--')
        plt.show()

    return slopes[1:]


def compute_csd(scores, labels):
    """
    Compute the cumulative sum diagram (CSD), i.e. a set of points P, such that:
    P_0 = (0, 0)
    P_i = (\sum_{j=1}^i w_j, \sum_{j=1}^i y'_j * w_j) for i = 1, ..., k'
    where
    k' = number of distinct values in scores s_i : i = 1 .. k
    s' = k' distinct values in scores s_i, i = 1 .. k, sorted in ascending order
    w_j = |{i : s_i = s'_j}| for j = 1 .. k' (i.e. number of times s'_j occurs in scores) 
    y'_j = \sum_{i  : s_i = s'_j} y_i / w_j, for j = 1 ... k'
        
    :param scores: a length-k vector of scores \in (-\inf, \inf)
    :param labels: a length-k vector of labels \in {0, 1}
    :return: (a vector of unique scores (length k', sorted), a set of CSD points P (k'+1,2))
    """

    # get unique values among the scores
    score_set = list(set(scores))
    # make sure they are still sorted
    score_set.sort()

    # get k', the number of distinct scores
    k_prime = len(score_set)
    # create intermediate vectors of lenght k' + 1 (to account for 0 point)
    weights = np.zeros(k_prime+1)
    y_prime = np.zeros(k_prime+1)
    points = np.zeros([k_prime+1, 2])

    # iterate through all unique scores
    for s_i, s in enumerate(score_set):
        # get the input points with the corresponding score
        indices = scores == s
        # indices here run from 1 .. k'
        j = s_i + 1
        # w_j = |# points with s_i == s'_j|, j = 1 .. k'
        weights[j] = np.sum(indices)
        # y'_j= \sum_{i : s_i == s'_j} y_i  / w_j, j = 1 .. k'
        y_prime[j] = np.sum(labels[indices]) / float(weights[j])
        # compute points P_i : i = 1 .. k'
        points[j, 0] = np.sum(weights[1:j+1])
        points[j, 1] = np.dot(weights[1:j+1], y_prime[1:j+1])

    return score_set, points, weights


def compute_gcm(points):
    """
    Compute the greatest convex minorant (GCM)
    :param points: Points P (k'+1, 2) that form the CSD (see above) 
    :return: (the set of points that define the GCM, the value of the GCM at the unique s'_j values)
    """
    # get the number of points in P
    n_points, _ = points.shape
    # find the point with the smallest second value
    min_point_index = np.argmin(points[:, 1])
    # recursively compute the GCM to the left and right
    print("left")
    left_gcm = compute_left_gcm(points, min_point_index)
    print("right")
    right_gcm = compute_right_gcm(points, min_point_index)
    # combine them to get the indices of GCM corner points
    gcm_corners = left_gcm[:-1] + right_gcm
    # extract the points that define the GCM
    gcm = points[gcm_corners, :]

    # compute the slope of the GCM at all points
    slopes = np.zeros(n_points)
    # iterate through the corners of the GCM
    for i, left_index in enumerate(gcm_corners[:-1]):
        right_index = gcm_corners[i+1]
        # compute the slope between adjacent points
        slope = (points[left_index, 1] - points[right_index, 1]) / float(points[left_index, 0] - points[right_index, 0])
        # record the value of the slope (to the left) at all intermediate points
        for j in range(left_index+1, right_index+1):
            slopes[j] = slope

    return gcm, slopes


def compute_left_gcm(points, index):
    """Recursively compute the GCM for points to the left of index"""
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
    """Recursively compute the GCM for points to the right of index"""
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
