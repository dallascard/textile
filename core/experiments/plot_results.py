import os
import re
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd

# import Agg to avoid network display problems
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



from ..util import file_handling as fh


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
    parser.add_option('--n_train', dest='n_train', default=None,
                      help='Train prop: default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=None,
                      help='Number of calibration instances: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
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
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='objective [f1|calibration]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')

    (options, args) = parser.parse_args()

    objective = options.objective
    cshift = options.cshift
    n_train = options.n_train
    n_calib = options.n_calib
    sample_labels = options.sample
    base = options.base
    subset = options.subset
    label = options.label
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    fig, ax = plt.subplots()

    if n_train is None:
        n_train = '*'
    else:
        n_train = str(int(n_train))

    if n_calib is None:
        n_calib = '*'
    else:
        n_calib = str(int(n_calib))

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    for objective_i, objective in enumerate(['f1', 'calibration']):
        basename = '*_' + label + '_*_' + model_type + '_' + penalty
        if model_type == 'MLP':
            basename += '_' + dh
        basename += '_' + n_train + '_' + n_calib + '_' + objective
        if model_type == 'MLP':
            basename += '_r?'
        if cshift is not None:
            basename += '_cshift'
        if sample_labels:
            basename += '_sampled'
        if base == 'mfc':
            basename += '_???_????_?'
        elif base == 'amazon':
            basename += '_????_?'

        print(basename)
        search_string = os.path.join('projects', base, subset, 'models', basename, 'results.csv')
        print(search_string)
        files = glob(search_string)
        files.sort()
        n_files = len(files)
        print(files)

        n_train_values = []
        n_calib_values = []
        for f in files:
            match = re.match(r'.*' + penalty + r'_([0-9]+)_([0-9]+)_*', f)
            n_train_values.append(int(match.group(1)))
            n_calib_values.append(int(match.group(2)))

        n_train_values = list(set(n_train_values))
        n_train_values.sort()
        print(n_train_values)

        n_calib_values = list(set(n_calib_values))
        n_calib_values.sort()
        print(n_calib_values)
        #if base == 'mfc':
        #    n_train_values = [100, 200, 400, 800]
        if base == 'amazon':
            n_calib_values = [100, 400, 1600, 3200]

        CC_nontrain = []
        PCC_nontrain = []
        SRS = []
        Venn = []
        CC_means = []
        PCC_means = []
        SRS_means = []
        Venn_means = []
        x = []
        n_train_means = []
        if n_train == '*':
            target_values = n_train_values
        elif n_calib == '*':
            target_values = n_calib_values
        for val in target_values:

            basename = '*_' + label + '_*_' + model_type + '_' + penalty
            if model_type == 'MLP':
                basename += '_' + dh
            if n_train == '*':
                n_train_val = val
                n_calib_val = n_calib
            elif n_calib == '*':
                n_train_val = n_train
                n_calib_val = val
            basename += '_' + str(n_train_val) + '_' + str(n_calib_val) + '_' + objective
            if model_type == 'MLP':
                basename += '_r?'
            if cshift is not None:
                basename += '_cshift'
            if sample_labels:
                basename += '_sampled'
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
            df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            mean_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            x.append(val)
            CC_nontrain.append(df.loc['CC_nontrain_averaged', 'RMSE'])
            PCC_nontrain.append(df.loc['PCC_nontrain_averaged', 'RMSE'])
            SRS.append(df.loc['calibration', 'RMSE'])
            Venn.append(df.loc['Venn_averaged', 'RMSE'])

            for f in files[1:]:
                print(f)
                results = fh.read_csv_to_df(f)
                df = results[['N', 'estimate', 'RMSE', 'contains_test']]
                mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']]
                x.append(val)
                CC_nontrain.append(df.loc['CC_nontrain_averaged', 'RMSE'])
                PCC_nontrain.append(df.loc['PCC_nontrain_averaged', 'RMSE'])
                SRS.append(df.loc['calibration', 'RMSE'])
                Venn.append(df.loc['Venn_averaged', 'RMSE'])

            mean_df = mean_df / float(n_files)

            n_train_means.append(int(val))
            CC_means.append(mean_df.loc['CC_nontrain_averaged', 'RMSE'])
            PCC_means.append(mean_df.loc['PCC_nontrain_averaged', 'RMSE'])
            SRS_means.append(mean_df.loc['calibration', 'RMSE'])
            Venn_means.append(mean_df.loc['Venn_averaged', 'RMSE'])

        print(n_train_means)
        print(CC_means)
        print(PCC_means)
        if objective == 'f1':
            colors = ['blue', 'orange', 'green', 'cyan']
            name = 'PCC acc'
        else:
            colors = ['black', 'green', 'red', 'yellow']
            name = 'PCC cal'

        if objective == 'f1':
            #ax.scatter(x, CC_nontrain, c=colors[0], alpha=0.5, s=10)
            ax.plot(n_train_means, CC_means, label='CC', alpha=0.5)

        #ax.scatter(x, PCC_nontrain, c=colors[1], alpha=0.5, s=10)
        ax.plot(n_train_means, PCC_means, label=name, alpha=0.5)

        if objective == 'f1':
            ax.plot(n_train_means, Venn_means,  label='Venn' + objective[:3], alpha=0.5)

        if objective == 'calibration':
            #ax.scatter(x, SRS, c=colors[2], alpha=0.5, s=10)
            ax.plot(n_train_means, SRS_means,  label='SRS', alpha=0.5)

    ax.legend()
    fig.savefig('test.pdf')

if __name__ == '__main__':
    main()
