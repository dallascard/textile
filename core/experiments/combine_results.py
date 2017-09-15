import os
from glob import glob
from optparse import OptionParser

import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective [f1|calibration]: default=%default')
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
    parser.add_option('-t', dest='train_prop', default=0.9,
                      help='Train prop: default=%default')
    parser.add_option('-p', dest='calib_prop', default=0.1,
                      help='Calibration prop: default=%default')
    parser.add_option('--base', dest='base', default='mfc',
                      help='base [mfc|amazon]: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='model type [LR|MLP]: default=%default')


    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    objective = options.objective
    cshift = options.cshift
    train_prop = str(float(options.train_prop))
    calib_prop = str(float(options.calib_prop))
    base = options.base
    model_type = options.model

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    basename = '*_' + model_type + '_l2_' + train_prop + '_' + calib_prop + '_' + objective
    if cshift is not None:
        basename += '_cshift'
    if base == 'mfc':
        basename += '_*_????_?'
    elif base == 'amazon':
        basename += '_????_?'

    print(basename)
    files = glob(os.path.join('projects', base, 'climate', 'models', basename, 'results.csv'))
    files.sort()
    n_files = len(files)

    print(files[0])
    results = fh.read_csv_to_df(files[0])
    df = pd.DataFrame(results[['estimate', 'RMSE', 'contains_test']].copy())

    for f in files[1:]:
        print(f)
        results = fh.read_csv_to_df(f)
        df += results[['estimate', 'RMSE', 'contains_test']]

    df = df / float(n_files)

    print(df)


if __name__ == '__main__':
    main()
