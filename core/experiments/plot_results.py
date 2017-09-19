import os
import re
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from ..util import file_handling as fh


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
    #parser.add_option('-t', dest='train_prop', default=0.9,
    #                  help='Train prop: default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=100,
                      help='Number of calibration items used: default=%default')
    parser.add_option('--base', dest='base', default='mfc',
                      help='base [mfc|amazon]: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='model type [LR|MLP]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='objective [f1|calibration]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')

    (options, args) = parser.parse_args()

    objective = options.objective
    cshift = options.cshift
    #n_calib = str(int(options.n_calib))
    n_calib = str(int(options.n_calib))
    base = options.base
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    for objective in ['f1', 'calibration']:
        basename = '*_' + model_type + '_' + penalty
        if model_type == 'MLP':
            basename += '_' + dh
        #basename += '_' + train_prop + '_' + calib_prop + '_' + objective
        #basename += '_*_' + objective
        basename += '_*_' + n_calib + '_' + objective
        if cshift is not None:
            basename += '_cshift'
        if base == 'mfc':
            basename += '_*_????_?'
        elif base == 'amazon':
            basename += '_????_?'

        print(basename)
        search_string = os.path.join('projects', base, '*', 'models', basename, 'results.csv')
        print(search_string)
        files = glob(search_string)
        files.sort()
        n_files = len(files)

        train_props = []
        for f in files:
            print(f)
            match = re.match(r'.*' + penalty + r'_([0-9]+\.[0-9]+)_.*', f)
            print(match.group(1))
            train_props.append(match.group(1))

        train_props = list(set(train_props))
        print(train_props)

        n_train = []
        CC_nontrain = []
        PCC_nontrain = []
        n_train_means = [0]
        CC_means = [0]
        PCC_means = [0]
        for t in train_props:
            basename = '*_' + model_type + '_' + penalty
            if model_type == 'MLP':
                basename += '_' + dh
            basename += '_' + t + '_' + n_calib + '_' + objective
            if cshift is not None:
                basename += '_cshift'
            if base == 'mfc':
                basename += '_*_????_?'
            elif base == 'amazon':
                basename += '_????_?'

            print(basename)
            files = glob(os.path.join('projects', base, '*', 'models', basename, 'results.csv'))
            files.sort()
            n_files = len(files)

            print(files[0])
            results = fh.read_csv_to_df(files[0])
            df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            mean_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            n_train.append(float(t))
            CC_nontrain.append(df.loc['CC_nontrain', 'RMSE'])
            PCC_nontrain.append(df.loc['PCC_nontrain', 'RMSE'])

            for f in files[1:]:
                print(f)
                results = fh.read_csv_to_df(f)
                df = results[['N', 'estimate', 'RMSE', 'contains_test']]
                mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']]
                n_train.append(float(t))
                CC_nontrain.append(df.loc['CC_nontrain', 'RMSE'])
                PCC_nontrain.append(df.loc['PCC_nontrain', 'RMSE'])

            mean_df = mean_df / float(n_files)

            n_train_means.append(float(t))
            CC_means.append(mean_df.loc['CC_nontrain', 'RMSE'])
            PCC_means.append(mean_df.loc['PCC_nontrain', 'RMSE'])

        print(n_train_means)
        print(CC_means)
        print(PCC_means)
        #plt.scatter(n_train, CC_nontrain)
        plt.scatter(n_train, PCC_nontrain, alpha=-0.5)
        #plt.scatter(n_train_means, CC_means)
        plt.scatter(n_train_means, PCC_means, alpha=0.5)
        plt.plot(np.array(n_train_means), np.array(PCC_means), alpha=0.5, label=objective)
    plt.legend()
    plt.savefig('test.pdf')
    #plt.show()

if __name__ == '__main__':
    main()
