from optparse import OptionParser

# module for Bayesian logistic regression
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression as lr

def main():

    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    # generate some simulated data
    n0 = 100
    n1 = 100
    p = 3
    mu1 = np.array([1, 1, 0])
    X0 = np.random.randn(n0, p)
    X1 = np.random.randn(n1, p) + mu1
    y0 = np.ones(n0)
    y1 = -1 * np.ones(n1)
    X = np.vstack([X0, X1])
    y = np.r_[y0, y1]

    model = lr(penalty='l2', fit_intercept=False)
    model.fit(X, y)
    print(model.coef_)

    print(X.shape)
    print(y.shape)
    m_0 = np.zeros(p)
    S_0 = np.eye(p)
    jj(X, y, m_0, S_0)


def jj(X, y, m_0, S_0, max_iter=2, tol=1e-6, z_init=None):
    # Jakkola and Jordan approximation
    # using notation in Bishop

    n, p = X.shape

    S_0_inv = np.linalg.inv(S_0)
    if z_init is None:
        z = np.random.randn(n) * 0.1
    else:
        z = z_init

    prev_bound = -np.inf

    for it in range(max_iter):
        lambda_z = lambda_func(z)
        S_N_inv = S_0_inv + 2 * np.dot(X.T, lambda_z.reshape(n, 1) * X)
        print(S_N_inv)

        # update variational expectation q(w) = N(m_N, S_N)
        S_N = np.linalg.inv(S_N_inv)
        print(S_N)
        m_N = np.dot(S_N_inv, np.dot(S_0_inv, m_0) + np.sum((y - 0.5).reshape((n, 1)) * X, axis=0))

        # update variational parameters
        inner_part = S_N + np.outer(m_N, m_N)
        for i in range(n):
            z[i] = np.sqrt(np.dot(X[i, :], np.dot(inner_part, X[i, :])))

        bound = compute_bound(z, m_0, S_0, S_0_inv, m_N, S_N, S_N_inv)
        delta = (bound - prev_bound) / float(prev_bound)
        prev_bound = bound

        print("%d %.2f %.4f" % (it, bound, delta))

        if delta < tol:
            break
    return m_N, S_N


def compute_bound(z, m_0, S_0, S_0_inv, m_N, S_N, S_N_inv):
    bound = 0.5 * np.log(np.linalg.det(S_N)) - np.log(np.linalg.det(S_0))
    bound += 0.5 * np.dot(m_N.T, np.dot(S_N_inv, m_N))
    bound -= 0.5 * np.dot(m_0.T, np.dot(S_0_inv, m_0))
    bound += np.sum(np.log(expit(z) - 0.5 * z + lambda_func(z) * z**2))
    return bound


def lambda_func(z):
    return -(expit(z) - 0.5) / (2*z)


if __name__ == '__main__':
    main()
