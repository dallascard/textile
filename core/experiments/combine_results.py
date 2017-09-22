import os
from glob import glob
from optparse import OptionParser

import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
    parser.add_option('-t', dest='train_prop', default=0.9,
                      help='Train prop: default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=100,
                      help='Number of calibration instances: default=%default')
    parser.add_option('--base', dest='base', default='mfc',
                      help='base [mfc|amazon]: default=%default')
    parser.add_option('--subset', dest='subset', default='*',
                      help='Subset of base (e.g. immigration: default=%default')
    parser.add_option('--label', dest='label', default='*',
                      help='Label (e.g. Economic: default=%default')
    #parser.add_option('--partition', dest='partition', default='*',
    #                  help='Partition for mfc (e.g. pre: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='model type [LR|MLP]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='objective [f1|calibration]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')


    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    objective = options.objective
    cshift = options.cshift
    train_prop = str(float(options.train_prop))
    n_calib = str(int(options.n_calib))
    base = options.base
    subset = options.subset
    label = options.label
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    basename = '*_' + label + '_*_' + model_type + '_' + penalty
    if model_type == 'MLP':
        basename += '_' + dh
    basename += '_' + train_prop + '_' + n_calib + '_' + objective
    if model_type == 'MLP':
        basename += '_r?'
    if cshift is not None:
        basename += '_cshift'
    if base == 'mfc':
        basename += '_???_????_?'
    elif base == 'amazon':
        basename += '_????_?'

    print(basename)
    files = glob(os.path.join('projects', base, subset, 'models', basename, 'results.csv'))
    files.sort()
    n_files = len(files)

    print(files[0])
    results = fh.read_csv_to_df(files[0])
    df = pd.DataFrame(results[['estimate', 'RMSE', 'contains_test']].copy())

    target_prop = results.loc['target', 'estimate']
    venn_av_lower = results.loc['Venn_averaged', '95lcl']
    venn_av_upper = results.loc['Venn_averaged', '95ucl']
    venn_outside_error = 0
    n_outside = 0

    if not (venn_av_lower < target_prop < venn_av_upper):
        venn_outside_error += max(venn_av_lower - target_prop, target_prop - venn_av_upper)
        n_outside += 1


    for f in files[1:]:
        print(f)
        results = fh.read_csv_to_df(f)
        df += results[['estimate', 'RMSE', 'contains_test']]

        target_prop = results.loc['target', 'estimate']
        venn_av_lower = results.loc['Venn_averaged', '95lcl']
        venn_av_upper = results.loc['Venn_averaged', '95ucl']
        venn_outside_error = 0
        n_outside = 0

        if not (venn_av_lower < target_prop < venn_av_upper):
            venn_outside_error += max(venn_av_lower - target_prop, target_prop - venn_av_upper)
            n_outside += 1


    df = df / float(n_files)
    if n_outside > 0:
        venn_outside_error /= float(n_outside)

    print(df)
    print("n_outside: %d" % n_outside)
    print("mean venn outside error = %0.6f" % venn_outside_error)

    # repeat for accuracy / f1
    files = glob(os.path.join('projects', base, subset, 'models', basename, 'accuracy.csv'))
    files.sort()
    n_files = len(files)

    print(files[0])
    results = fh.read_csv_to_df(files[0])
    df = results.copy()

    for f in files[1:]:
        print(f)
        results = fh.read_csv_to_df(f)
        df += results

    df = df / float(n_files)
    print(df)


if __name__ == '__main__':
    main()
