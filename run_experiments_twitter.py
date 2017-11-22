import os
from optparse import OptionParser

from core.experiments import combo
from core.experiments import over_time_split_and_fit


def main():
    usage = "%prog project config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--cshift', action="store_true", dest="cshift", default=False,
                      help='Covariate shift method [None|classify]: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    config = args[1]

    objective = options.objective
    penalty = options.penalty
    cshift = options.cshift

    for n_train in [1000, 2250, 5000, 10000, 22500]:
        print("\n\nStarting n_train = %d", n_train)
        if n_train == 1000:
            n_repeats = 10
        else:
            n_repeats = 5
        over_time_split_and_fit.test_over_time(project, 'train', config, 'LR', 'month', 6, 6, n_train, 0, penalty, loss='log', objective=objective, do_ensemble=True, label='positive', intercept=True, n_dev_folds=5, repeats=n_repeats, cshift=cshift)


if __name__ == '__main__':
    main()
