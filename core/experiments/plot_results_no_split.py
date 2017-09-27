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
import seaborn


from ..util import file_handling as fh


def main():
    usage = "%prog "
    parser = OptionParser(usage=usage)
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

    (options, args) = parser.parse_args()

    sampled = options.sample
    base = options.base
    subset = options.subset
    label = options.label
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    fig, ax = plt.subplots()

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    for objective_i, objective in enumerate(['f1', 'calibration']):
        n_train = '*'
        basename = '*_' + label + '_*_' + model_type + '_' + penalty
        if model_type == 'MLP':
            basename += '_' + dh
        basename += '_' + n_train + '_' + objective
        if sampled:
            basename += '_sampled'
        basename += '_nosplit_?'

        print(basename)
        search_string = os.path.join('projects', base, subset, 'models', basename, 'results.csv')
        print(search_string)
        files = glob(search_string)
        files.sort()
        n_files = len(files)
        print(files)

        n_train_values = []
        for f in files:
            match = re.match(r'.*' + penalty + r'_([0-9]+)_*', f)
            n_train_values.append(int(match.group(1)))

        n_train_values = list(set(n_train_values))
        n_train_values.sort()
        print(n_train_values)
        if base == 'mfc':
            n_train_values = [100, 200, 400, 800, 1600]
        elif base == 'amazon':
            n_train_values = [200, 800, 3200, 6400]

        ACC_nontrain = []
        CC_nontrain = []
        PCC_nontrain = []
        SRS = []
        Venn = []

        ACC_means = []
        CC_means = []
        PCC_means = []
        SRS_means = []
        Venn_means = []

        CC_stds = []
        PCC_stds = []
        SRS_stds = []
        Venn_stds = []

        x = []
        n_train_means = []
        for n_train in n_train_values:
            basename = '*_' + label + '_*_' + model_type + '_' + penalty
            if model_type == 'MLP':
                basename += '_' + dh
            basename += '_' + str(n_train) + '_' + objective
            if sampled:
                basename += '_sampled'
            basename += '_nosplit_?'

            print(basename)
            files = glob(os.path.join('projects', base, subset, 'models', basename, 'results.csv'))
            files.sort()
            n_files = len(files)

            print(files[0])
            results = fh.read_csv_to_df(files[0])
            df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            mean_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            sq_mean_df = pd.DataFrame(mean_df.values ** 2, index=mean_df.index, columns=mean_df.columns)
            max_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            x.append(n_train)
            #n_train_means.append(n_train)
            #n_train.append(df.loc['train', 'N'])
            ACC_nontrain.append(df.loc['ACC_internal', 'RMSE'])
            CC_nontrain.append(df.loc['CC_nontrain_averaged', 'RMSE'])
            PCC_nontrain.append(df.loc['PCC_nontrain_averaged', 'RMSE'])
            SRS.append(df.loc['train', 'RMSE'])
            Venn.append(df.loc['Venn_internal_averaged', 'RMSE'])

            for f in files[1:]:
                print(f)
                results = fh.read_csv_to_df(f)
                df = results[['N', 'estimate', 'RMSE', 'contains_test']]
                mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']]
                sq_mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']].values ** 2
                max_df = np.maximum(max_df, results[['N', 'estimate', 'RMSE', 'contains_test']])
                x.append(n_train)
                #n_train.append(float(t))
                #n_train.append(df.loc['train', 'N'])
                ACC_nontrain.append(df.loc['ACC_internal', 'RMSE'])
                CC_nontrain.append(df.loc['CC_nontrain_averaged', 'RMSE'])
                PCC_nontrain.append(df.loc['PCC_nontrain_averaged', 'RMSE'])
                SRS.append(df.loc['train', 'RMSE'])
                Venn.append(df.loc['Venn_internal_averaged', 'RMSE'])

            mean_df = mean_df / float(n_files)
            sq_mean_df = sq_mean_df / float(n_files)

            n_train_means.append(int(n_train))
            #n_train_means.append(mean_df.loc['train', 'N'])
            ACC_means.append(mean_df.loc['ACC_internal', 'RMSE'])
            CC_means.append(mean_df.loc['CC_nontrain_averaged', 'RMSE'])
            PCC_means.append(mean_df.loc['PCC_nontrain_averaged', 'RMSE'])
            SRS_means.append(mean_df.loc['train', 'RMSE'])
            Venn_means.append(mean_df.loc['Venn_internal_averaged', 'RMSE'])

            CC_stds.append(np.sqrt(sq_mean_df.loc['CC_nontrain_averaged', 'RMSE'] - mean_df.loc['CC_nontrain_averaged', 'RMSE']**2))
            PCC_stds.append(np.sqrt(sq_mean_df.loc['PCC_nontrain_averaged', 'RMSE'] - mean_df.loc['PCC_nontrain_averaged', 'RMSE']**2))
            SRS_stds.append(np.sqrt(sq_mean_df.loc['train', 'RMSE'] - mean_df.loc['train', 'RMSE']**2))


        print(n_train_means)
        print(CC_means)
        print(PCC_means)
        CB6 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']

        dot_size = 5
        linewidth=2
        if objective == 'f1':
            #ax.scatter(np.array(x)-36, ACC_nontrain, c=CB6[0], alpha=0.5, s=dot_size)
            #ax.plot(n_train_means, ACC_means, label='ACC', c=CB6[0], linewidth=linewidth)

            ax.scatter(np.array(x)-18, CC_nontrain, c=CB6[1], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, CC_means, label='CC', c=CB6[1], linewidth=linewidth)
            #ax.plot(n_train_means, np.array(CC_means) + np.array(CC_stds), linestyle='dashed', c=colors[0], label='CC (+1std)', alpha=0.5)

        #ax.scatter(x, PCC_nontrain, c=colors[1], alpha=0.5, s=10)
        if objective == 'f1':
            ax.scatter(np.array(x), PCC_nontrain, c=CB6[2], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, PCC_means, label='PCC (acc)', c=CB6[2], linewidth=linewidth)
        else:
            ax.scatter(np.array(x)+18, PCC_nontrain, c=CB6[3], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, PCC_means, label='PCC (cal)', c=CB6[3], linewidth=linewidth)
        #ax.plot(n_train_means, np.array(PCC_means) + np.array(PCC_stds), label=name + ' (+1std)', linestyle='dashed', alpha=0.5, c=colors[1])
        #ax.plot(n_train_means, PCC_maxes, label=name + ' (max)', linestyle='dashed', alpha=0.5)

        #ax.plot(n_train_means, Venn_means,  label='Venn' + objective[:3], alpha=0.5)

        if objective == 'calibration':
            ax.scatter(np.array(x)+36, SRS, c=CB6[4], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, SRS_means,  label='SRS', c=CB6[4], linewidth=linewidth)
            #ax.plot(n_train_means, np.array(SRS_means) + np.array(SRS_stds), label='SRS' + ' (+1std)', linestyle='dashed', alpha=0.5, c=colors[2])

    ax.set_xlabel('Number of training instances (L)')
    ax.set_ylabel('Mean absolute error')
    ax.set_ylim(-0.01, 0.35)

    ax.legend()
    fig.savefig('test.pdf')

if __name__ == '__main__':
    main()
