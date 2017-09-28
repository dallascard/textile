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
    usage = "%prog output_file.pdf"
    parser = OptionParser(usage=usage)
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
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
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')
    parser.add_option('--offset', dest='offset', default=9,
                      help='Offset: default=%default')
    parser.add_option('--output', dest='output', default='acc',
                      help='Output [f1|acc]: default=%default')


    (options, args) = parser.parse_args()

    output_file = args[0]

    cshift = options.cshift
    n_calib = options.n_calib
    sample_labels = options.sample
    base = options.base
    subset = options.subset
    label = options.label
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))
    offset = int(options.offset)
    output = options.output

    fig, ax = plt.subplots()

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
        basename += '_' + '*' + '_' + n_calib + '_' + objective
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
        search_string = os.path.join('projects', base, subset, 'models', basename, 'accuracy.csv')
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
        if base == 'mfc':
            n_train_values = [100, 200, 400, 800, 1600]
        if base == 'amazon':
            n_train_values = [400, 800, 1600, 3200]

        train = []
        ACC = []
        CC = []
        PCC_nontrain = []
        SRS = []
        Venn = []

        train_means = []
        ACC_means = []
        CC_means = []
        PCC_means = []
        SRS_means = []
        Venn_means = []

        x = []
        n_train_means = []

        target_values = n_train_values

        for val in target_values:

            basename = '*_' + label + '_*_' + model_type + '_' + penalty
            if model_type == 'MLP':
                basename += '_' + dh
            n_train_val = val
            n_calib_val = n_calib
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
            files = glob(os.path.join('projects', base, subset, 'models', basename, 'accuracy.csv'))
            files.sort()
            n_files = len(files)

            print(files[0])
            results = fh.read_csv_to_df(files[0])
            mean_df = pd.DataFrame(results[['f1', 'acc']].copy())
            #df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            #mean_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            #max_df = pd.DataFrame(results[['N', 'estimate', 'RMSE', 'contains_test']].copy())
            x.append(val)
            PCC_nontrain.append(results.loc['test', output])

            #ACC.append(df.loc['ACC', 'RMSE'])
            #CC.append(df.loc['CC_nontrain', 'RMSE'])
            #PCC_nontrain.append(df.loc['PCC_nontrain', 'RMSE'])
            #SRS.append(df.loc['calibration', 'RMSE'])
            #Venn.append(df.loc['Venn_internal', 'RMSE'])

            for f in files[1:]:
                print(f)
                results = fh.read_csv_to_df(f)
                mean_df += pd.DataFrame(results[['f1', 'acc']].copy())
                PCC_nontrain.append(results.loc['test', output])
                x.append(val)

                #ACC.append(df.loc['ACC', 'RMSE'])
                #CC.append(df.loc['CC_nontrain', 'RMSE'])
                #PCC_nontrain.append(df.loc['PCC_nontrain', 'RMSE'])
                #SRS.append(df.loc['calibration', 'RMSE'])
                #Venn.append(df.loc['Venn_internal', 'RMSE'])

            mean_df = mean_df / float(n_files)

            n_train_means.append(int(val))
            #train_means.append(mean_df.loc['train', 'RMSE'])
            #ACC_means.append(mean_df.loc['ACC', 'RMSE'])
            #CC_means.append(mean_df.loc['CC_nontrain', 'RMSE'])
            PCC_means.append(mean_df.loc['test', output])
            #SRS_means.append(mean_df.loc['calibration', 'RMSE'])
            #Venn_means.append(mean_df.loc['Venn_internal', 'RMSE'])

            #ACC_maxes.append(max_df.loc['ACC', 'RMSE'])
            #SRS_maxes.append(max_df.loc['calibration', 'RMSE'])
            #Venn_maxes.append(max_df.loc['Venn', 'RMSE'])
            #PCC_maxes.append(max_df.loc['PCC_nontrain', 'RMSE'])


        CB6 = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02']
        dot_size = 5
        linewidth = 2

        if objective == 'f1':
            ax.scatter(np.array(x)-0.5*offset, PCC_nontrain, c=CB6[2], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, PCC_means, label='PCC (acc)', c=CB6[2], linewidth=linewidth)
        else:
            ax.scatter(np.array(x)+0.5*offset, PCC_nontrain, c=CB6[3], alpha=0.5, s=dot_size)
            ax.plot(n_train_means, PCC_means, label='PCC (cal)', c=CB6[3], linewidth=linewidth)

    ax.set_xlabel('Number of training instances (L)')
    if output == 'acc':
        ax.set_ylabel('Classification accuracy on target corpus')
    else:
        ax.set_ylabel('Classification f1')
    #if base == 'mfc':
    #    ax.set_ylim(-0.01, 0.4)
    #else:
    #    ax.set_ylim(-0.01, 0.15)

    if base == 'mfc':
        ax.set_title('MFC')
    else:
        ax.set_title('Amazon')

    #ax.legend(loc='upper right')
    ax.legend()
    fig.savefig(output_file, bbox_inches='tight')


if __name__ == '__main__':
    main()
