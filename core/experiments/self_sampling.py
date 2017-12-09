from optparse import OptionParser

import numpy as np
from scipy.special import expit
from scipy import sparse

from sklearn.linear_model import Ridge


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    N = 1000
    sample_maes = np.zeros(N)
    model_maes = np.zeros(N)
    for i in range(N):
        sample_mae, model_mae = do_experiment(px=0.01, p=5000, sample_size=500, n=10000)
        sample_maes[i] = sample_mae
        model_maes[i] = model_mae
        if (i+1) % 10 == 0:
            print(i+1)
    print(np.mean(sample_maes), np.mean(model_maes))


def do_experiment(px, p, sample_size, n=10000, w_bias=0.0):
    weights = np.random.randn(p) + w_bias
    bias = np.random.randn()
    X = sparse.csr_matrix(sparse.random(n, p, density=px) > 0, dtype=int)
    y = expit(X.dot(weights) + bias)
    py = np.mean(y)

    sample = np.random.choice(np.arange(n), size=sample_size, replace=False)
    sample_X = X[sample, :]
    sample_y = y[sample]

    model = Ridge()
    model.fit(sample_X, sample_y)
    pred = model.predict(X)
    pred_mean = np.mean(pred)
    sample_mean = np.mean(sample_y)

    sample_mae = np.abs(py - sample_mean)
    model_mae = np.abs(py - pred_mean)
    return sample_mae, model_mae






if __name__ == '__main__':
    main()
