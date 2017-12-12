import os
import re
from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--intrinsic', action="store_true", dest="intrinsic", default=False,
                      help='Output figure for intrinsic data: default=%default')

    (options, args) = parser.parse_args()

    intrinsic = options.intrinsic

    base_dir = 'results'
    if intrinsic:
        csv1 = os.path.join(base_dir, 'city_500.csv')
        csv2 = os.path.join(base_dir, 'city_5000.csv')
        csv3 = os.path.join(base_dir, 'twitter_500.csv')
        csv4 = os.path.join(base_dir, 'twitter_5000.csv')
    else:
        csv1 = os.path.join(base_dir, 'mfc_500.csv')
        csv2 = os.path.join(base_dir, 'mfc_2000.csv')
        csv3 = os.path.join(base_dir, 'help_500.csv')
        csv4 = os.path.join(base_dir, 'help_5000.csv')

    df1 = pd.read_csv(csv1, index_col=0, header=0)
    df2 = pd.read_csv(csv2, index_col=0, header=0)
    df3 = pd.read_csv(csv3, index_col=0, header=0)
    df4 = pd.read_csv(csv4, index_col=0, header=0)
    dfs = [df1, df2, df3, df4]

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    fig.subplots_adjust(wspace=0)

    names = ['Train', 'CC', 'PCC(acc)', 'PCC(F1)', 'Platt', 'Reweighting', 'ACC', 'PCC(cal)', ]
    y = list(range(len(names)))

    y.reverse()

    if intrinsic:
        datasets = ['MAE: Yelp (500)', 'MAE: Yelp (5000)', 'MAE: Twitter (500)', 'MAE: Twitter (5000)']
    else:
        datasets = ['MAE: MFC (500)', 'MAE: MFC (2000)', 'MAE: Amazon (500)', 'MAE: Amazon (5000)']

    for d_i, dataset in enumerate(datasets):
        ax = axes[d_i]
        df = dfs[d_i]
        ax.barh(y, df.loc[names].mean(axis=1), alpha=0.6, facecolor='blue')
        for y_i, y_val in enumerate(y):
            vals = df.loc[names[y_i]]
            ax.scatter(vals, np.ones_like(vals) * y_val, s=10, color='k', alpha=0.5)
        ax.set_xlim(0, 0.29)
        ax.set_xlabel(dataset)
        if d_i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(names)

    if intrinsic:
        plt.savefig('/Users/dcard/Submissions/NAACL2018/fig_intrinsic.pdf', bbox_inches='tight')
    else:
        plt.savefig('/Users/dcard/Submissions/NAACL2018/fig_extrinsic.pdf', bbox_inches='tight')

if __name__ == '__main__':
    main()
