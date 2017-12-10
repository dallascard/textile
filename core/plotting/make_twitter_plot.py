import os
import re
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
    parser.add_option('--prefix', dest='prefix', default='test',
                      help='Output prefix (optional): default=%default')

    (options, args) = parser.parse_args()

    output = options.prefix

    targets = ['0.45', '0.5', '0.525', '0.55', '0.575', '0.6', '0.625', '0.65', '0.675', '0.7', '0.75']

    ACC_values = {}
    PCC_values = {}
    for t in targets:
        ACC_values[t] = []
        PCC_values[t] = []

    for t in targets:
        files = glob(os.path.join('projects', 'twitter', 'models', 'train_positive_l1_f1_5000_0' + t + '_167-167_*', 'results.csv'))

        for f in files:
            print(f)
            df_f = fh.read_csv_to_df(f)
            ACC_values[t].append(df_f['MAE'].loc['ACC_internal'])
            PCC_values[t].append(df_f['MAE'].loc['PCC'])

    ACC_means = np.zeros(len(targets))
    PCC_means = np.zeros(len(targets))
    for t_i, t in enumerate(targets):
        ACC_means[t_i] = np.mean(ACC_values[t])
        PCC_means[t_i] = np.mean(PCC_values[t])

    fig, ax = plt.subplots()
    x = [float(t) for t in targets]
    ax.plot(x, ACC_means, 'b-', alpha=0.8)
    ax.plot(x, PCC_means, 'g--', alpha=0.8)
    for t in targets:
        ACC_vals = ACC_values[t]
        PCC_vals = PCC_values[t]
        x = np.ones(len(ACC_vals)) * t
        ax.scatter(x-0.01, ACC_vals, s=10, color='b', alpha=0.5)
        ax.scatter(x+0.01, PCC_vals, s=10, color='g', alpha=0.5)

    plt.savefig(output + '.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
