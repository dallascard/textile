import random
from optparse import OptionParser

import numpy as np
from scipy.special import expit, gammaln
from sklearn.linear_model import LogisticRegression as lr


def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--max_iter', dest='max_iter', default=50,
                      help='Maximum number of iterations: default=%default')
    parser.add_option('--tol', dest='tol', default=1e-6,
                      help='Convergence tolerance: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()
    max_iter = int(options.max_iter)
    tol = float(options.tol)

    # generate some simulated data
    n0 = 100
    n1 = 100
    p = 20
    #mu0 = np.random.randn(p)
    mu0 = np.zeros(p)
    mu0[0] = 2.0
    mu1 = -mu0
    mu1[0] = 4.0
    #mu1 = np.zeros(p)
    #mu1[0] = 2.0
    X0 = np.random.randn(n0, p) + mu0
    X1 = np.random.randn(n1, p) + mu1
    y0 = np.ones(n0)
    y1 = np.zeros(n1)
    X = np.vstack([X0, X1])
    y = np.r_[y0, y1]

    X1 = add_ones_col(X)

    print("MLE with separate intercept")
    model = lr(penalty='l2', C=1.0, fit_intercept=True, intercept_scaling=100)
    model.fit(X, y)
    print(model.intercept_)
    print(model.coef_)

    print("basic with separate intercept")
    w, V_n = batch_basic(X, y, alpha_0=1.0, add_intercept=True, max_iter=max_iter, tol=tol)
    print(w[0])
    print(w[1:])

    print("multilevel with intercept")
    w, V_n, E_alpha = batch_multilevel(X, y, add_intercept=True, max_iter=max_iter, tol=tol)
    print(w[0])
    print(w[1:])
    print(E_alpha[:4])

    print("multilevel ARD with intercept")
    w, V_n, E_alphas = batch_multilevel_ard(X, y, add_intercept=True, max_iter=max_iter, tol=tol)
    print(w[0])
    print(w[1:])
    print(E_alphas[0:4])





# TODO: add posterior predictive

# now obsolete
"""
def batch_multilevel(X, y, s_0=1e-2, r_0=1e-4, add_intercept=False, max_iter=50, tol=1e-6):

    if add_intercept:
        X = add_ones_col(X)

    n, d = X.shape

    # compute things that won't change
    sum_yX_over_2 = np.sum((y - 0.5).reshape(n, 1) * X, axis=0)
    s_n = s_0 + d / 2.0
    bound_const = -gammaln(s_0) + s_0 * np.log(r_0) + gammaln(s_n) + s_n

    # initialize lambda(z) to 1/8
    z = np.zeros(n)
    lambda_z = np.ones(n) / 8.0
    # initialize E[alpha] to s_0 / r_0
    E_alpha = s_0 / float(r_0)

    # compute initial V_n and w values
    inv_V_n = E_alpha * np.eye(d) + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
    V_n = np.linalg.inv(inv_V_n)
    w = np.dot(V_n, sum_yX_over_2)

    # compute an initial bound
    r_n = r_0 + 0.5 * (np.dot(w, w) + np.trace(V_n))
    prev_bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const)
    #prev_bound = -np.inf

    # iteratively update parameters
    for it in range(max_iter):
        z = np.sqrt(np.sum(np.dot(X, V_n + np.outer(w, w)) * X, axis=1))
        lambda_z = lambda_func(z)

        r_n = r_0 + 0.5 * (np.dot(w, w) + np.trace(V_n))
        E_alpha = s_n / float(r_n)

        inv_V_n = E_alpha * np.eye(d) + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
        V_n = np.linalg.inv(inv_V_n)
        w = np.dot(V_n, sum_yX_over_2)

        bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const)

        delta = (bound - prev_bound) / np.abs(float(prev_bound))
        prev_bound = bound

        print("%d %.2f %.6f" % (it, bound, delta))
        if delta < tol and it > 1:
            break

    return w, V_n, E_alpha
"""


def compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const):
    log_det_sign, log_abs_det = np.linalg.slogdet(inv_V_n)
    assert log_det_sign > 0
    log_det_V_n = -log_abs_det

    bound = bound_const + 0.5 * log_det_V_n
    bound += 0.5 * np.dot(w, np.dot(inv_V_n, w))
    bound += np.sum(-np.log(1 + np.exp(-z))) - 0.5 * np.sum(z) + np.sum(lambda_z * z**2)
    bound -= r_0 * np.sum(s_n / r_n)
    bound -= np.sum(s_n * np.log(r_n))
    return bound


def batch_multilevel_ard(X, y, add_intercept=False, s_0=1e-2, r_0=1e-4, max_iter=50, tol=1e-6):
    # Do automatic relevance determination (hopefully this handles the intercept okay...)

    if add_intercept:
        # since we are using independent variance terms for each parameter, we can treat the intercept as a feature
        X = add_ones_col(X)

    n, d = X.shape

    # compute things that won't change
    sum_yX_over_2 = np.sum((y - 0.5).reshape(n, 1) * X, axis=0)
    s_n = s_0 + 0.5
    bound_const = d * (-gammaln(s_0) + s_0 * np.log(r_0) + (gammaln(s_n) + s_n))

    # initialize lambda(z) to 1/8
    z = np.zeros(n)
    lambda_z = np.ones(n) / 8.0
    # initialize E[alpha] to s_0 / r_0
    E_alpha = np.ones(d) * s_0 / float(r_0)

    # compute initial V_n and w values
    inv_V_n = np.diag(E_alpha) + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
    V_n = np.linalg.inv(inv_V_n)
    w = np.dot(V_n, sum_yX_over_2)

    # compute an initial bound
    r_n = r_0 + 0.5 * (w ** 2 + np.diag(V_n))
    prev_bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const)
    #prev_bound = -np.inf

    # iteratively update parameters
    for it in range(max_iter):
        z = np.sqrt(np.sum(np.dot(X, V_n + np.outer(w, w)) * X, axis=1))
        lambda_z = lambda_func(z)

        r_n = r_0 + 0.5 * (w ** 2 + np.diag(V_n))
        E_alpha = s_n / r_n

        inv_V_n = np.diag(E_alpha) + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
        V_n = np.linalg.inv(inv_V_n)
        w = np.dot(V_n, sum_yX_over_2)

        bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const)

        delta = (bound - prev_bound) / np.abs(float(prev_bound))
        prev_bound = bound

        print("%d %.2f %.6f" % (it, bound, delta))
        if delta < tol and it > 1:
            break

    return w, V_n, E_alpha


def batch_multilevel(X, y, add_intercept=False, s_0=1e-2, r_0=1e-4, max_iter=50, tol=1e-6):
    # Basic multilevel logistic regression (but with a separate parameter for intercept variance)

    n, d = X.shape
    d_aug = d

    if add_intercept:
        # since we are using independent variance terms for each parameter, we can treat the intercept as a feature
        X = add_ones_col(X)
        d_aug += 1

    # compute things that won't change
    sum_yX_over_2 = np.sum((y - 0.5).reshape(n, 1) * X, axis=0)
    s_n = s_0 + d / 2.0
    s_n_intercept = s_0 + 0.5
    bound_const = -gammaln(s_0) + s_0 * np.log(r_0) + gammaln(s_n) + s_n
    if add_intercept:
        bound_const += gammaln(s_n_intercept) + s_n_intercept

    # initialize lambda(z) to 1/8
    z = np.zeros(n)
    lambda_z = np.ones(n) / 8.0
    # initialize E[alpha] to s_0 / r_0
    E_alpha = s_0 / float(r_0)

    # compute initial V_n and w values
    inv_V_n = E_alpha * np.eye(d_aug) + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
    if add_intercept:
        E_alpha_intercept = s_n_intercept / float(r_0)
        inv_V_n[0, 0] = E_alpha_intercept
    V_n = np.linalg.inv(inv_V_n)
    w = np.dot(V_n, sum_yX_over_2)

    # compute an initial bound
    if add_intercept:
        # treat the intercept rate separately
        r_n_intercept = r_0 + 0.5 * (w[0] ** 2 + V_n[0, 0])
        r_n = r_0 + 0.5 * (np.dot(w[1:], w[1:]) + np.trace(V_n[1:, 1:]))
        s_n_both = np.r_[s_n, s_n_intercept]
        r_n_both = np.r_[r_n, r_n_intercept]
        prev_bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n_both, r_n_both, bound_const)
    else:
        r_n = r_0 + 0.5 * (np.dot(w, w) + np.trace(V_n))
        prev_bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const)
    #prev_bound = -np.inf

    # iteratively update parameters
    for it in range(max_iter):
        z = np.sqrt(np.sum(np.dot(X, V_n + np.outer(w, w)) * X, axis=1))
        lambda_z = lambda_func(z)

        if add_intercept:
            # treat the intercept rate separately
            r_n_intercept = r_0 + 0.5 * (w[0] ** 2 + V_n[0, 0])
            r_n = r_0 + 0.5 * (np.dot(w[1:], w[1:]) + np.trace(V_n[1:, 1:]))
        else:
            r_n = r_0 + 0.5 * (np.dot(w, w) + np.trace(V_n))
        E_alpha = s_n / r_n

        inv_V_n = E_alpha * np.eye(d_aug)
        if add_intercept:
            E_alpha_intercept = s_n_intercept / float(r_n_intercept)
            inv_V_n[0, 0] = E_alpha_intercept
        inv_V_n += 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
        V_n = np.linalg.inv(inv_V_n)
        w = np.dot(V_n, sum_yX_over_2)

        if add_intercept:
            s_n_both = np.r_[s_n, s_n_intercept]
            r_n_both = np.r_[r_n, r_n_intercept]
            bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n_both, r_n_both, bound_const)
        else:
            bound = compute_bound_multilevel(z, lambda_z, w, inv_V_n, r_0, s_n, r_n, bound_const)

        delta = (bound - prev_bound) / np.abs(float(prev_bound))
        prev_bound = bound

        print("%d %.2f %.6f" % (it, bound, delta))
        if delta < tol and it > 1:
            break

    E_alphas = E_alpha * np.ones(d_aug)
    if add_intercept:
        E_alphas[0] = E_alpha_intercept
    return w, V_n, E_alphas


def batch_multilevel_predictive(X, y, V, inv_V):
    pass


def batch_basic(X, y, alpha_0, add_intercept=False, alpha_intercept=0.01, max_iter=50, tol=1e-6):
    """
    Estimate a posterior over weights in a logistic regression model using variational inference
     and the sigmoid approximation from Jakkola and Jordan, and notation from Jan Drugowitsch 
    :param X: n x d data matrix
    :param y: length-n vector of data labels (binary only): {0, 1}
    :param alpha_0: fixed value of diagonal precision matrix for Normal prior on weights
    :param add_intercept: if True, add an intercept to the data and fit a parameter for it
    :param alpha_intercept: uninformative alpha parameter for the intercept if add_intercept is True 
    :param max_iter: maximum number of iterations
    :param tol: relative change required for convergence 
    :return: Estimated mean (length-d) and covariance matrix (d x d) for posterior on weights
    """

    if add_intercept:
        X = add_ones_col(X)

    n, d = X.shape

    # compute things that don't change
    sum_yX_over_2 = np.sum((y - 0.5).reshape(n, 1) * X, axis=0)
    alpha = alpha_0 * np.ones(d)
    # if using an intercept, use low shrinkage on the corresponding parameter
    if add_intercept:
        alpha[0] = alpha_intercept
    inv_V_0 = np.diag(alpha)

    # initialize lambda(z) to 1/8
    z = np.zeros(n)
    lambda_z = np.ones(n) / 8.0

    # compute initial V_n and w values
    inv_V_n = inv_V_0 + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
    V_n = np.linalg.inv(inv_V_n)
    w = np.dot(V_n, sum_yX_over_2)

    # compute the initial value of the bound
    prev_bound = compute_bound_basic(z, lambda_z, alpha, w, inv_V_n)

    # repeat until convergence
    for it in range(max_iter):
        # update variational parameters for logit
        z = np.sqrt(np.sum(np.dot(X, V_n + np.outer(w, w)) * X, axis=1))
        lambda_z = lambda_func(z)

        # udpate expected mean and variance of weights
        inv_V_n = inv_V_0 + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
        V_n = np.linalg.inv(inv_V_n)
        w = np.dot(V_n, sum_yX_over_2)

        # compute bound and check convergence
        bound = compute_bound_basic(z, lambda_z, alpha, w, inv_V_n)
        delta = (bound - prev_bound) / np.abs(float(prev_bound))
        prev_bound = bound

        print("%d %.2f %.6f" % (it, bound, delta))

        if delta < tol:
            break
    return w, V_n


def compute_bound_basic(z, lambda_z, alpha, w, inv_V_n):
    log_det_sign, log_abs_det = np.linalg.slogdet(inv_V_n)
    assert log_det_sign > 0
    log_det_V_n = -log_abs_det

    d = len(w)
    # log_det(diag(A)) = \sum_i log(a_ii)
    bound = 0.5 * (log_det_V_n + np.sum(np.log(alpha)))
    bound += 0.5 * np.dot(w.T, np.dot(inv_V_n, w))
    bound += np.sum(np.log(expit(z)) - 0.5 * z + lambda_z * z**2)
    return bound


def iterative_basic(X, y, alpha_0, add_intercept=False, max_iter=50, tol=1e-6, verbose=False):
    """ Ported more or less directly from Drugowitsch's Matlab code"""

    # note that this does not currently support using a separate alpha parameter for the intercept
    if add_intercept:
        X = add_ones_col(X)

    n, d = X.shape

    # initialize parameters
    alpha = alpha_0 * np.ones(d)

    V = np.diag(1.0/alpha)
    inv_V = np.diag(alpha)
    log_det_V = -np.sum(np.log(alpha))
    w = np.zeros(d)

    order = list(range(n))

    for epoch in range(10):
        random.shuffle(order)
        # iterate over all data
        for i in order:
            x = X[i, :]

            Vx = np.dot(V, x)
            VxVx = np.outer(Vx, Vx)
            c = np.dot(x, Vx)
            xx = np.outer(x, x)
            t_w = np.dot(inv_V, w) + (y[i] - 0.5) * x

            V_z = V - VxVx / (4.0 + c)
            inv_V_z = inv_V + xx / 4.0
            log_det_V_z = log_det_V - np.log(1.0 + c / 4.0)
            w = np.dot(V_z, t_w)

            prev_bound = 0.5 * (log_det_V_z * np.dot(w, np.dot(inv_V_z, w))) - np.log(2)

            for it in range(max_iter):
                z = np.sqrt(np.dot(x, np.dot(V_z + np.outer(w, w), x)))
                lambda_z = lambda_func(z)

                V_z = V - (2 * lambda_z / (1.0 + 2.0 * lambda_z * c)) * VxVx
                inv_V_z = inv_V + 2 * lambda_z * xx
                log_det_V_z = log_det_V - np.log(1.0 + 2.0 * lambda_z * c)
                w = np.dot(V_z, t_w)

                bound = 0.5 * (log_det_V_z + np.dot(w, np.dot(inv_V_z, w)) - z) - np.log(1.0 + np.exp(-z)) + lambda_z * z**2

                if prev_bound > bound:
                    print("bound moving in wrong direction")

                delta = (bound - prev_bound) / np.abs(float(prev_bound))
                prev_bound = bound

                if delta < tol:
                    break
            if verbose:
                print("%d %d %.2f %.6f" % (i, it, bound, delta))

            V = V_z
            inv_V = inv_V_z
            log_det_V = log_det_V_z

        print(w[0])
    return w, V


def lambda_func(z):
    return 0.5 * (expit(z) - 0.5) / z


def add_ones_col(X):
    return np.hstack((np.ones([X.shape[0], 1]), X))

if __name__ == '__main__':
    main()
