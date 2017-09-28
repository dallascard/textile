import os
import re
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy.stats import levene

# import Agg to avoid network display problems
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt



from ..util import file_handling as fh


def main():
    usage = "%prog output_file.pdf"
    parser = OptionParser(usage=usage)
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
    parser.add_option('--n_train', dest='n_train', default=100,
                      help='Train prop: default=%default')
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
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')
    parser.add_option('--offset', dest='offset', default=8,
                      help='offset: default=%default')
    parser.add_option('--averaged', action="store_true", dest="averaged", default=False,
                      help='Use value of averaged with calib: default=%default')

    (options, args) = parser.parse_args()
    output_file = args[0]

    cshift = options.cshift
    n_train = options.n_train
    sample_labels = options.sample
    base = options.base
    subset = options.subset
    label = options.label
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))
    offset = int(options.offset)
    averaged = options.averaged

    if averaged:
        venn_target = 'Venn_averaged'
    else:
        venn_target = 'Venn'

    fig, ax = plt.subplots()

    n_train = str(int(n_train))

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    for objective_i, objective in enumerate(['calibration', 'f1']):
        basename = '*_' + label + '_*_' + model_type + '_' + penalty
        if model_type == 'MLP':
            basename += '_' + dh
        basename += '_' + n_train + '_' + '*' + '_' + objective
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
            if model_type == 'LR':
                match = re.match(r'.*' + penalty + r'_([0-9]+)_([0-9]+)_*', f)
            else:
                match = re.match(r'.*' + penalty + '_' + dh + r'_([0-9]+)_([0-9]+)_*', f)
            n_train_values.append(int(match.group(1)))
            n_calib_values.append(int(match.group(2)))

        n_train_values = list(set(n_train_values))
        n_train_values.sort()
        print(n_train_values)

        n_calib_values = list(set(n_calib_values))
        n_calib_values.sort()
        print(n_calib_values)

        if base == 'amazon':
            n_calib_values = [50, 100, 200, 400, 800]

        #CC_nontrain = []
        PCC_nontrain = []
        SRS = []
        Venn = []

        PCC_means = []
        SRS_means = []
        Venn_means = []

        PCC_stds = []
        SRS_stds = []
        Venn_stds = []

        SRS_values = {}
        Venn_values = {}

        x = []
        n_train_means = []
        target_values = n_calib_values

        for val in target_values:
            SRS_values[val] = []
            Venn_values[val] = []
            basename = '*_' + label + '_*_' + model_type + '_' + penalty
            if model_type == 'MLP':
                basename += '_' + dh
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
            sq_mean_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].values ** 2, index=mean_df.index, columns=mean_df.columns)
            max_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            x.append(val)
            PCC_nontrain.append(df.loc['PCC_nontrain', 'RMSE'])
            SRS.append(df.loc['calibration', 'RMSE'])
            Venn.append(df.loc[venn_target, 'RMSE'])
            SRS_values[val].append(df.loc['calibration', 'RMSE'])
            Venn_values[val].append(df.loc[venn_target, 'RMSE'])

            for f in files[1:]:
                print(f)
                results = fh.read_csv_to_df(f)
                df = results[['N', 'estimate', 'RMSE', 'contains_test']]
                mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']]
                sq_mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']].values ** 2
                max_df = np.maximum(max_df, results[['N', 'estimate', 'RMSE', 'contains_test']].values)
                x.append(val)
                PCC_nontrain.append(df.loc['PCC_nontrain', 'RMSE'])
                SRS.append(df.loc['calibration', 'RMSE'])
                Venn.append(df.loc[venn_target, 'RMSE'])
                SRS_values[val].append(df.loc['calibration', 'RMSE'])
                Venn_values[val].append(df.loc[venn_target, 'RMSE'])

            mean_df = mean_df / float(n_files)
            sq_mean_df = sq_mean_df / float(n_files)

            n_train_means.append(int(val))
            PCC_means.append(mean_df.loc['PCC_nontrain', 'RMSE'])
            SRS_means.append(mean_df.loc['calibration', 'RMSE'])
            Venn_means.append(mean_df.loc[venn_target, 'RMSE'])

            PCC_stds.append(np.sqrt(sq_mean_df.loc['PCC_nontrain', 'RMSE'] - mean_df.loc['PCC_nontrain', 'RMSE'] ** 2))
            SRS_stds.append(np.sqrt(sq_mean_df.loc['calibration', 'RMSE'] - mean_df.loc['calibration', 'RMSE'] ** 2))
            Venn_stds.append(np.sqrt(sq_mean_df.loc[venn_target, 'RMSE'] - mean_df.loc[venn_target, 'RMSE'] ** 2))

        print(n_train_means)
        print(PCC_means)
        CB6 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
        dot_size = 5
        linewidth = 2

        #if objective == 'f1':
        #    ax.scatter(np.array(x)-offset, PCC_nontrain, c=CB6[2], alpha=0.5, s=dot_size)
        #    ax.plot([np.min(n_train_means), np.max(n_train_means)], [np.mean(PCC_means), np.mean(PCC_means)], label='PCC (acc)', c=CB6[2], linewidth=linewidth, linestyle='dashed')

        if objective == 'calibration':
            ax.scatter(np.array(x)-offset, PCC_nontrain, c=CB6[3], alpha=0.5, s=dot_size)
            ax.plot([np.min(n_train_means), np.max(n_train_means)], [np.mean(PCC_means), np.mean(PCC_means)], label='PCC (cal)', c=CB6[3], linewidth=linewidth, linestyle='dashed')

        if objective == 'f1':
            ax.scatter(np.array(x), SRS, c=CB6[4], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, SRS_means,  label='SRS', c=CB6[4], linewidth=linewidth)
            #ax.plot(n_train_means, np.array(SRS_means) + np.array(SRS_stds),  label='SRS', c=CB6[4], linestyle='dashed')

            ax.scatter(np.array(x)+offset, Venn, c=CB6[5], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, Venn_means,  label='IVAP', c=CB6[5], linewidth=linewidth)
            #ax.plot(n_train_means, np.array(Venn_means) + np.array(Venn_stds),  label='SRS', c=CB6[5], linestyle='dashed')

            for val in target_values:
                print(val, levene(SRS_values[val], Venn_values[val], center='median'))

    ax.set_xlabel('Number of calibration instances (C)')
    ax.set_ylabel('Mean absolute error')
    if base == 'mfc':
        ax.set_ylim(-0.01, 0.4)
    else:
        ax.set_ylim(-0.01, 0.15)

    if base == 'mfc':
        ax.set_title('MFC')
    else:
        ax.set_title('Amazon')

    ax.legend()
    fig.savefig(output_file, bbox_inches='tight')

if __name__ == '__main__':
    main()
