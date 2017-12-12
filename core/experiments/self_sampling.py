from optparse import OptionParser

import numpy as np
from scipy.special import expit
from scipy import sparse
from scipy.stats import ttest_rel, wilcoxon
from sklearn.linear_model import LogisticRegression

from ..main.train import train_lr_model_with_cv

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n', default=10000,
                      help='Total instances: default=%default')
    parser.add_option('-p', dest='p', default=500,
                      help='number of features: default=%default')
    parser.add_option('-s', dest='sample_size', default=500,
                      help='Number of instances to sample: default=%default')
    parser.add_option('--dx', dest='dx', default=0.01,
                      help='Density of features in X: default=%default')
    parser.add_option('--dw', dest='dw', default=0.1,
                      help='Density of weights: default=%default')
    parser.add_option('--iter', dest='iter', default=100,
                      help='Number of iterations: default=%default')


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
    print("paired t-test:", ttest_rel(sample_maes, model_maes))
    print("Wilcoxon:", wilcoxon(sample_maes, model_maes))


def do_experiment(n, p, sample_size, px, pw, w_bias=0.0):
    weights = np.random.randn(p) + w_bias
    ablation = np.array(np.random.rand(p) >= pw, dtype=int)
    weights = weights * ablation
    bias = np.random.randn()
    X = sparse.csr_matrix(sparse.random(n, p, density=px) > 0, dtype=int)
    y = expit(X.dot(weights) + bias)
    py = np.mean(y)

    sample = np.random.choice(np.arange(n), size=sample_size, replace=False)
    sample_X = X[sample, :]
    sample_y = y[sample]

    model = LogisticRegression(penalty='l1')
    model.fit(sample_X, sample_y)
    pred = model.predict(X)
    pred_mean = np.mean(pred)
    sample_mean = np.mean(sample_y)

    sample_mae = np.abs(py - sample_mean)
    model_mae = np.abs(py - pred_mean)

    return sample_mae, model_mae






if __name__ == '__main__':
    main()
