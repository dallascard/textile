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
    parser.add_option('--first_test_year', dest='first_year', default=2011,
                      help='Use training data from before this year: default=%default')
    parser.add_option('--last_test_year', dest='last_year', default=2012,
                      help='Last year of test data to use: default=%default')
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
                      help='Model type [LR|MLP]: default=%default')
    parser.add_option('--lower', dest='lower', default=None,
                      help='Lower bound to enforce positive weights: default=%default')
    parser.add_option('--suffix', dest='suffix', default='',
                      help='Suffix to mdoel name: default=%default')


    (options, args) = parser.parse_args()
    project = args[0]
    config = args[1]

    n_train = options.n_train
    if n_train is not None:
        n_train = int(n_train)
    n_calib = int(options.n_calib)
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

    pairs = [('sources', 'Atlanta_Journal_and_Constitution'),
             ('sources', 'Denver_Post'),
             ('sources', 'Herald-Sun'),
             ('sources', 'NY_Daily_News'),
             ('sources', 'NY_Times'),
             ('sources', 'Palm_Beach_Post'),
             ('sources', 'Philadelphia_Inquirer'),
             ('sources', 'San_Jose_Mercury_News'),
             ('sources', 'St._Louis_Post-Dispatch'),
             ('sources', 'St._Paul_Pioneer_Press'),
             ('sources', 'Tampa_Bay_Times'),
             ('sources', 'USA_Today'),
             ('sources', 'Washington_Post')
             ]

    for subset, label in pairs:
        print("\n\nStarting", subset, label)
        try:
            over_time_split_and_fit.test_over_time(project, subset, config, model_type, 'year', first_year, last_year, n_train, n_calib, penalty, suffix, loss='log', objective=objective, do_ensemble=True, label=label, intercept=True, sample_labels=sample_labels, n_dev_folds=5, list_size=ls, repeats=repeats, lower=lower, cshift=cshift)
        except:
            print("Failed on %s" % label)

if __name__ == '__main__':
    main()
