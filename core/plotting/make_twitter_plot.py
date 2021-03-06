import os
from glob import glob
from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--prefix', dest='prefix', default='twitter_ACC',
                      help='Output prefix (optional): default=%default')

    (options, args) = parser.parse_args()

    output = options.prefix

    #targets = ['0.5', '0.55', '0.575', '0.6', '0.625', '0.65', '0.7']
    targets = ['0.425', '0.525', '0.6', '0.625', '0.65', '0.725', '0.825']

    ACC_values = {}
    PCC_values = {}
    CC_values = {}
    for t in targets:
        ACC_values[t] = []
        PCC_values[t] = []
        CC_values[t] = []

    for t in targets:
        files = glob(os.path.join('projects', 'twitter', 'models', 'train_positive_l1_f1_5000_0' + t + '_165-165_*', 'results.csv'))

        for f in files:
            print(f)
            df_f = fh.read_csv_to_df(f)
            ACC_values[t].append(df_f['MAE'].loc['ACC_internal'])
            PCC_values[t].append(df_f['MAE'].loc['PCC'])
            CC_values[t].append(df_f['MAE'].loc['CC'])

    ACC_means = np.zeros(len(targets))
    PCC_means = np.zeros(len(targets))
    CC_means = np.zeros(len(targets))
    for t_i, t in enumerate(targets):
        ACC_means[t_i] = np.mean(ACC_values[t])
        PCC_means[t_i] = np.mean(PCC_values[t])
        CC_means[t_i] = np.mean(CC_values[t])

    fig, ax = plt.subplots(figsize=(5, 2))
    x = [float(t) for t in targets]
    ax.plot(x, ACC_means, 'b-', alpha=0.9)
    ax.plot(x, PCC_means, 'g-', alpha=0.9)
    #ax.plot(x, CC_means, 'r-', alpha=0.9)
    for t_i, t in enumerate(targets):
        ACC_vals = ACC_values[t]
        PCC_vals = PCC_values[t]
        CC_vals = CC_values[t]
        n_vals = len(ACC_vals)
        x = np.ones(n_vals) * float(t)
        if t_i == 0:
            labels = [r'PCC$^{\mathrm{F}_1}$', 'ACC', 'CC']
        else:
            labels = [None, None, None]
        ax.scatter(x, PCC_vals, s=10, color='g', marker='x', alpha=0.6, label=labels[0])
        ax.scatter(x, ACC_vals, s=10, color='b', alpha=0.6, label=labels[1])
        #ax.scatter(x, CC_vals, s=10, color='r', marker='+', alpha=0.6, label=labels[2])

    ax.legend(loc='upper center')
    ax.set_xlabel('Modified target label proportion')
    ax.set_ylabel('AE')
    #ax.set_ylim(0, 0.1)
    #x = [float(t) for t in targets]
    #x_vals = [0.5, 0.55, 0.6, 0.65, 0.7]
    #ax.set_xticks(x_vals)
    #ax.set_xticklabels(labels)
    plt.savefig(output + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
