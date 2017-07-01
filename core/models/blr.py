from optparse import OptionParser

# module for Bayesian logistic regression
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as lr
import matplotlib.pyplot as plt

def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--max_iter', dest='max_iter', default=5,
                      help='Maximum number of iterations: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()
    max_iter = int(options.max_iter)

    # generate some simulated data
    n0 = 100
    n1 = 100
    p = 20
    mu0 = np.random.randn(p)
    mu1 = -mu0
    X0 = np.random.randn(n0, p) + mu0
    X1 = np.random.randn(n1, p) + mu1
    y0 = np.ones(n0)
    #y1 = -1 * np.ones(n1)
    y1 = np.zeros(n1)
    X = np.vstack([X0, X1])
    y = np.r_[y0, y1]


    model = lr(penalty='l2', fit_intercept=False)
    model.fit(X, y)
    print(model.coef_)

    #fig, ax = plt.subplots()
    #ax.scatter(X0[:, 0], X0[:, 1])
    #ax.scatter(X1[:, 0], X1[:, 1])
    #plt.show()

    m_0 = np.zeros(p)
    S_0 = np.eye(p)
    m_N, S_N = jj(X, y, m_0, S_0, max_iter=max_iter)
    print(m_N)


def jj(X, y, m_0, S_0, max_iter=5, tol=1e-6, z_init=None):
    # Jakkola and Jordan approximation
    # using notation in Bishop

    n, p = X.shape

    S_0_inv = np.linalg.inv(S_0)
    if z_init is None:
        z = np.random.rand(n) + 1e-4
    else:
        z = z_init

    prev_bound = -np.inf

    for it in range(max_iter):
        lambda_z = lambda_func(z)
        S_N_inv = S_0_inv + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)

        # update variational expectation q(w) = N(m_N, S_N)
        S_N = np.linalg.inv(S_N_inv)
        m_N = np.dot(S_N, np.dot(S_0_inv, m_0) + np.sum((y - 0.5).reshape((n, 1)) * X, axis=0))

        # update variational parameters
        inner_part = S_N + np.outer(m_N, m_N)
        z_sq = np.zeros(n)
        for i in range(n):
            z_sq[i] = np.dot(X[i, :], np.dot(inner_part, X[i, :]))
        assert np.all(z_sq >= 0)
        z = np.sqrt(z_sq)


        bound = compute_bound(z, m_0, S_0, S_0_inv, m_N, S_N, S_N_inv)
        delta = (bound - prev_bound) / np.abs(float(prev_bound))
        prev_bound = bound

        print("%d %.2f %.4f" % (it, bound, delta))

        if delta < tol:
            break
    return m_N, S_N


def compute_bound(z, m_0, S_0, S_0_inv, m_N, S_N, S_N_inv):
    bound = 0.5 * np.log(np.linalg.det(S_N))
    bound -= np.log(np.linalg.det(S_0))
    bound += 0.5 * np.dot(m_N.T, np.dot(S_N_inv, m_N))
    bound -= 0.5 * np.dot(m_0.T, np.dot(S_0_inv, m_0))
    bound += np.sum(np.log(expit(z)) - 0.5 * z + lambda_func(z) * z**2)
    return bound


def lambda_func(z):
    return (expit(z) - 0.5) / (2.0*z)


if __name__ == '__main__':
    main()
