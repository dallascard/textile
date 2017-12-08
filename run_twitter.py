from optparse import OptionParser

from core.experiments import over_time_split_and_fit2


def main():
    usage = "%prog project config.json"
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default=None,
                      help='Number of training instances to use (0 for all): default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=0,
                      help='Number of test instances to use for calibration: default=%default')
    #parser.add_option('--train_start', dest='train_start', default=None,
    #                  help='Start of training range (before test start if None): default=%default')
    #parser.add_option('--train_end', dest='train_end', default=None,
    #                  help='end of trainign range (before test start if None): default=%default')
    parser.add_option('--first_test_day', dest='first_day', default=107,
                      help='Use training data from before this year: default=%default')
    parser.add_option('--last_test_day', dest='last_day', default=130,
                      help='Last year of test data to use: default=%default')
    parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
                      help='Minimum value of training hyperparameter: default=%default')
    parser.add_option('--alpha_max', dest='alpha_max', default=1000,
                      help='Maximum value of training hyperparameter: default=%default')
    parser.add_option('--n_alphas', dest='n_alphas', default=8,
                      help='Number of alpha values to try: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [calibration|f1]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--cshift', action="store_true", dest="cshift", default=False,
                      help='Covariate shift method [None|classify]: default=%default')
    parser.add_option('--n_cshift', dest='n_cshift', default=None,
                      help='Number of data points to use for covariate shift model: default=%default')
    parser.add_option('-r', dest='repeats', default=1,
                      help='Repeats: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|DAN]: default=%default')
    parser.add_option('--dh', dest='dh', default=300,
                      help='Hidden layer size for MLP [0 for None]: default=%default')
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
    first_day = int(options.first_day)
    last_day = int(options.last_day)
    objective = options.objective
    penalty = options.penalty
    cshift = options.cshift
    n_cshift = options.n_cshift
    if n_cshift is not None:
        n_cshift = int(n_cshift)
    repeats = int(options.repeats)
    model_type = options.model
    suffix = options.suffix

    alpha_min = float(options.alpha_min)
    alpha_max = float(options.alpha_max)
    n_alphas = int(options.n_alphas)

    dh = int(options.dh)
    dropout = float(options.dropout)
    init_lr = float(options.init_lr)
    patience = int(options.patience)
    max_epochs = int(options.max_epochs)

    seed = 42
    for day in range(first_day, last_day+1):
        print("\n\nStarting", day)
        over_time_split_and_fit2.test_over_time(project, 'train', config, model_type, 'dayofyear', train_start=None, train_end=None, test_start=day, test_end=day, n_train=n_train, n_calib=n_calib, penalty=penalty, suffix=suffix, seed=seed, alpha_min=alpha_min, alpha_max=alpha_max, n_alphas=n_alphas, loss='log', objective=objective, do_ensemble=True, label='positive', intercept=True, n_dev_folds=5, repeats=repeats, cshift=cshift, n_cshift=n_cshift, dh=dh, init_lr=init_lr, dropout=dropout, patience=patience, max_epochs=max_epochs, min_test=1000)


if __name__ == '__main__':
    main()
