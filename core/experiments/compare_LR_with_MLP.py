import os
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default=100,
                      help='Train prop: default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=100,
                      help='Number of calibration instances: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--base', dest='base', default='mfc',
                      help='base [mfc|amazon]: default=%default')
    parser.add_option('--subset', dest='subset', default='*',
                      help='Subset of base (e.g. immigration: default=%default')
    parser.add_option('--label', dest='label', default='*',
                      help='Label (e.g. Economic: default=%default')
    parser.add_option('--partition', dest='partition', default=None,
                      help='Partition for mfc (e.g. pre: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='objective [f1|calibration]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')


    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    objective = options.objective
    n_train = str(int(options.n_train))
    n_calib = str(int(options.n_calib))
    sample_labels = options.sample
    base = options.base
    subset = options.subset
    label = options.label
    partition = options.partition
    penalty = options.penalty
    dh = str(int(options.dh))


    cv_f1s = []
    test_f1s = []
    PCC_maes = []

    n_LR = 0
    n_MLP = 0

    for model_type in ['LR', 'MLP']:
        basename = '*_' + label + '_*_' + model_type + '_' + penalty
        if model_type == 'MLP':
            basename += '_' + dh
        basename += '_' + n_train + '_' + n_calib + '_' + objective
        if model_type == 'MLP' and base == 'mfc':
            basename += '_r?'
        if sample_labels:
            basename += '_sampled'
        if base == 'mfc':
            if partition is None:
                basename += '_???_????_?'
            else:
                basename += '_' + partition + '_?'
        elif base == 'amazon':
            if partition is None:
                basename += '_????_?'
            else:
                basename += '_' + partition + '_?'

        print(basename)
        files = glob(os.path.join('projects', base, subset, 'models', basename, 'results.csv'))
        files.sort()
        n_files = len(files)

        if model_type == 'LR':
            n_LR += 1
        else:
            n_MLP += 1

        print(files[0])
        results = fh.read_csv_to_df(files[0])
        df = pd.DataFrame(columns=['estimate', 'MAE', 'MSE', 'contains_test', 'max_MAE'])
        #df = pd.DataFrame(results[['estimate', 'RMSE', 'contains_test']].copy())
        target_estimate = results.loc['target', 'estimate']
        for loc in results.index:
            df.loc[loc, 'estimate'] = results.loc[loc, 'estimate']
            df.loc[loc, 'MAE'] = results.loc[loc, 'RMSE']
            df.loc[loc, 'contains_test'] = results.loc[loc, 'contains_test']
            df.loc[loc, 'MSE'] = (results.loc[loc, 'estimate'] - target_estimate) ** 2
            df.loc[loc, 'max_MAE'] = results.loc[loc, 'RMSE']

        PCC_maes.append(results.loc['PCC_nontrain', 'RMSE'])

        file_dir, _ = os.path.split(files[0])
        accuracy_file = os.path.join(file_dir, 'accuracy.csv')
        accuracy_df = fh.read_csv_to_df(accuracy_file)

        cv_f1s.append(accuracy_df.loc['cross_val', 'f1'])
        test_f1s.append(accuracy_df.loc['test', 'f1'])

        for f in files[1:21]:
            print(f)
            results = fh.read_csv_to_df(f)
            #df += results[['estimate', 'RMSE', 'contains_test']]

            if model_type == 'LR':
                n_LR += 1
            else:
                n_MLP += 1

            target_estimate = results.loc['target', 'estimate']
            for loc in results.index:
                df.loc[loc, 'estimate'] += results.loc[loc, 'estimate']
                df.loc[loc, 'MAE'] += results.loc[loc, 'RMSE']
                df.loc[loc, 'contains_test'] += results.loc[loc, 'contains_test']
                df.loc[loc, 'MSE'] += (results.loc[loc, 'estimate'] - target_estimate) ** 2
                df.loc[loc, 'max_MAE'] = max(df.loc[loc, 'max_MAE'], results.loc[loc, 'RMSE'])

            PCC_maes.append(results.loc['PCC_nontrain', 'RMSE'])

            file_dir, _ = os.path.split(f)
            accuracy_file = os.path.join(file_dir, 'accuracy.csv')
            accuracy_df = fh.read_csv_to_df(accuracy_file)

            cv_f1s.append(accuracy_df.loc['cross_val', 'f1'])
            test_f1s.append(accuracy_df.loc['test', 'f1'])

    print(len(cv_f1s))
    print(n_LR)
    print(n_MLP)
    print(np.mean(cv_f1s[:n_LR]), np.mean(cv_f1s[n_LR:]))
    print(ttest_rel(cv_f1s[:n_LR], cv_f1s[n_LR:]))
    print(np.mean(test_f1s[:n_LR]), np.mean(test_f1s[n_LR:]))
    print(ttest_rel(test_f1s[:n_LR], test_f1s[n_LR:]))
    print(np.mean(PCC_maes[:n_LR]), np.mean(PCC_maes[n_LR:]))
    print(ttest_rel(PCC_maes[:n_LR], PCC_maes[n_LR:]))


if __name__ == '__main__':
    main()
