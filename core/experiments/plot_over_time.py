import os
import glob
from optparse import OptionParser

from ..experiments import over_time
from ..util import file_handling as fh
from ..util import dirs

# import Agg to avoid network display problems
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():
    usage = "%prog logfile.json"
    parser = OptionParser(usage=usage)
    #parser.add_option('--n_train', dest='n_train', default=100,
    #                  help='Number of training instances to use (0 for all): default=%default')
    #parser.add_option('--n_calib', dest='n_calib', default=100,
    #                  help='Number of test instances to use for calibration: default=%default')
    #parser.add_option('--sample', action="store_true", dest="sample", default=False,
    #                  help='Sample labels instead of averaging: default=%default')
    #parser.add_option('--suffix', dest='suffix', default='',
    #                  help='Suffix to mdoel name: default=%default')
    #parser.add_option('--model', dest='model', default='LR',
    #                  help='Model type [LR|MLP]: default=%default')
    #parser.add_option('--dh', dest='dh', default=100,
    #                  help='Hidden layer size for MLP [0 for None]: default=%default')
    #parser.add_option('--label', dest='label', default='label',
    #                  help='Label name: default=%default')
    #parser.add_option('--cshift', dest='cshift', default=None,
    #                  help='Covariate shift method [None|classify]: default=%default')
    #parser.add_option('--penalty', dest='penalty', default='l2',
    #                  help='Regularization type: default=%default')
    #parser.add_option('--no_intercept', action="store_true", dest="no_intercept", default=False,
    #                  help='Use to fit a model with no intercept: default=%default')
    #parser.add_option('--objective', dest='objective', default='f1',
    #                  help='Objective for choosing best alpha [calibration|f1]: default=%default')
    #parser.add_option('--verbose', action="store_true", dest="verbose", default=False,
    #                  help='Print more output: default=%default')

    (options, args) = parser.parse_args()

    #project_dir = args[0]
    #subset = args[1]
    logfile = args[0]

    #n_train = int(options.n_train)
    #n_calib = int(options.n_calib)
    #sample_labels = options.sample
    #suffix = options.suffix
    #model_type = options.model
    #dh = int(options.dh)
    #label = options.label
    #penalty = options.penalty
    #cshift = options.cshift
    #objective = options.objective
    #intercept = not options.no_intercept
    #repeats = int(options.repeats)
    #verbose = options.verbose

    plot_over_time(logfile)


def plot_over_time(logfile):
    log = fh.read_json(logfile)
    model_basename = over_time.make_model_basename(log)

    project_dir = log['project']

    files = glob.glob(os.path.join(dirs.dir_models(project_dir), model_basename + '_????', 'results.csv'))
    print(files)

    n_train_values = []
    pcc_values = []
    for f in files:
        df = fh.read_csv_to_df(f)
        n_train_values.append(df.loc['PCC_test', 'N'])
        pcc_values.append(df.loc['PCC_test', 'MAE'])

    print(n_train_values)
    print(pcc_values)

    #fig, ax = plt.subplots()
    #ax.scatter(n_train_values, pcc_values, label='all')


    """
    files = glob.glob(os.path.join(dirs.dir_models(project_dir), model_basename + '_????', 'results.csv'))
    print(files)

    n_train_values = []
    pcc_values = []
    for f in files:
        df = fh.read_csv_to_df(f)
        n_train_values.append(df.loc['PCC_test', 'N'])
        pcc_values.append(df.loc['PCC_test', 'MAE'])

    ax.scatter(n_train_values, pcc_values, label='fight')
    ax.legend()
    fig.savefig('test.pdf', bbox_inches='tight')
    """


if __name__ == '__main__':
    main()
