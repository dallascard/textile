import os
from optparse import OptionParser

from core.experiments import combo
from core.experiments import over_time_split_and_fit


def main():
    usage = "%prog project config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default=None,
                      help='Number of training instances to use (0 for all): default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=0,
                      help='Number of test instances to use for calibration: default=%default')
    parser.add_option('--train_start', dest='train_start', default=None,
                      help='Start of training range (before test start if None): default=%default')
    parser.add_option('--train_end', dest='train_end', default=None,
                      help='end of trainign range (before test start if None): default=%default')
    parser.add_option('--first_test_year', dest='first_year', default=2011,
                      help='Use training data from before this year: default=%default')
    parser.add_option('--last_test_year', dest='last_year', default=2012,
                      help='Last year of test data to use: default=%default')
    parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
                      help='Minimum value of training hyperparameter: default=%default')
    parser.add_option('--alpha_max', dest='alpha_max', default=1000,
                      help='Maximum value of training hyperparameter: default=%default')
    parser.add_option('--n_alphas', dest='n_alphas', default=8,
                      help='Number of alpha values to try: default=%default')
    parser.add_option('--ls', dest='ls', default=10,
                      help='List size (for DL): default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--cshift', action="store_true", dest="cshift", default=False,
                      help='Covariate shift method [None|classify]: default=%default')
    parser.add_option('-r', dest='repeats', default=3,
                      help='Repeats: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|DAN]: default=%default')
    parser.add_option('--dh', dest='dh', default=300,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
    parser.add_option('--lower', dest='lower', default=None,
                      help='Lower bound to enforce positive weights: default=%default')
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to mdoel name: default=%default')
    parser.add_option('--dropout', dest='dropout', default=0.0,
                      help='Apply word dropout to DANs: default=%default')
    parser.add_option('--lr', dest='init_lr', default=0.01,
                      help='Initial learning rate for DAN training: default=%default')
    parser.add_option('--patience', dest='patience', default=5,
                      help='Patience for DAN training: default=%default')
    parser.add_option('--max_epochs', dest='max_epochs', default=50,
                      help='Maximum number of epochs for DAN training: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    config = args[1]

    n_train = options.n_train
    if n_train is not None:
        n_train = int(n_train)
    n_calib = int(options.n_calib)
    train_start = options.train_start
    if train_start is not None:
        train_start = int(train_start)
    train_end = options.train_end
    if train_end is not None:
        train_end = int(train_end)
    first_year = int(options.first_year)
    last_year = int(options.last_year)
    ls = int(options.ls)
    objective = options.objective
    sample_labels = options.sample
    penalty = options.penalty
    cshift = options.cshift
    repeats = int(options.repeats)
    model_type = options.model
    lower = options.lower
    if lower is not None:
        lower = float(lower)
    suffix = options.suffix

    alpha_min = float(options.alpha_min)
    alpha_max = float(options.alpha_max)
    n_alphas = int(options.n_alphas)

    dh = int(options.dh)
    dropout = float(options.dropout)
    init_lr = float(options.init_lr)
    patience = int(options.patience)
    max_epochs = int(options.max_epochs)

    pairs = [('framing', 'Economic'), ('framing', 'Legality'), ('framing', 'Health'), ('framing', 'Political'), ('framing', 'Capacity'), ('framing', 'Crime')]

    seed = 42
    for subset, label in pairs:
        print("\n\nStarting", subset, label)
        over_time_split_and_fit.test_over_time(project, subset, config, model_type, 'year', train_start, train_end, first_year, last_year, n_train, n_calib, penalty, suffix, seed=seed, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, loss='log', objective=objective, do_ensemble=True, label=label, intercept=True, sample_labels=sample_labels, n_dev_folds=5, list_size=ls, repeats=repeats, lower=lower, cshift=cshift, dh=dh, init_lr=init_lr, dropout=dropout, patience=patience, max_epochs=max_epochs)
        seed += 1


if __name__ == '__main__':
    main()
