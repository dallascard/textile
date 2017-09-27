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

        CC_nontrain = []
        PCC_nontrain = []
        n_train_means = [0]
        CC_means = []
        PCC_means = []
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
            x.append(n_train)
            #n_train_means.append(n_train)
            #n_train.append(df.loc['train', 'N'])
            CC_nontrain.append(df.loc['CC_nontrain_averaged', 'RMSE'])
            PCC_nontrain.append(df.loc['PCC_nontrain_averaged', 'RMSE'])

            for f in files[1:]:
                print(f)
                results = fh.read_csv_to_df(f)
                df = results[['N', 'estimate', 'RMSE', 'contains_test']]
                mean_df += results[['N', 'estimate', 'RMSE', 'contains_test']]
                x.append(n_train)
                #n_train.append(float(t))
                #n_train.append(df.loc['train', 'N'])
                CC_nontrain.append(df.loc['CC_nontrain_averaged', 'RMSE'])
                PCC_nontrain.append(df.loc['PCC_nontrain_averaged', 'RMSE'])

            mean_df = mean_df / float(n_files)

            n_train_means.append(int(n_train))
            #n_train_means.append(mean_df.loc['train', 'N'])
            CC_means.append(mean_df.loc['CC_nontrain_averaged', 'RMSE'])
            PCC_means.append(mean_df.loc['PCC_nontrain_averaged', 'RMSE'])

        print(n_train_means)
        print(CC_means)
        print(PCC_means)
        if objective == 'f1':
            colors = ['blue', 'orange']
            name = 'PCC acc'
        else:
            colors = ['green', 'magenta']
            name = 'PCC cal'

        if objective == 'f1':
            ax.scatter(x, CC_nontrain, c=colors[0], alpha=0.5, s=10)
            ax.scatter(n_train_means, CC_means, c=colors[0], alpha=0.5, s=20)
            ax.plot(n_train_means, CC_means, c=colors[0], label='CC', alpha=0.5)

        ax.scatter(x, PCC_nontrain, c=colors[1], alpha=0.5, s=10)
        ax.scatter(n_train_means, PCC_means, c=colors[1], alpha=0.5, s=20)
        ax.plot(n_train_means, PCC_means, c=colors[1], label=name, alpha=0.5)
        #plt.plot(np.array(n_train_means), np.array(PCC_means), alpha=0.5, label=objective)

    ax.legend()
    fig.savefig('test.pdf')
    #plt.show()

if __name__ == '__main__':
    main()
