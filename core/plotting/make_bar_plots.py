import os
import re
from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from ..util import file_handling as fh


def main():
    usage = "%prog csv_results_files(f1_cshift)"
    parser = OptionParser(usage=usage)
    parser.add_option('--prefix', dest='prefix', default='test',
                      help='Output prefix (optional): default=%default')
    parser.add_option('--similar', action="store_true", dest="similar", default=False,
                      help='Only use the most similar examples: default=%default')
    parser.add_option('--different', action="store_true", dest="different", default=False,
                      help='Only use the most different examples: default=%default')
    parser.add_option('--balanced', action="store_true", dest="balanced", default=False,
                      help='Only use the most balanced examples: default=%default')
    parser.add_option('--unbalanced', action="store_true", dest="unbalanced", default=False,
                      help='Only use the most unbalanced examples: default=%default')
    parser.add_option('-p', dest='percentile', default=50,
                      help='percentile (optional): default=%default')
    parser.add_option('--twitter', action="store_true", dest="twitter", default=False,
                      help='Special for twitter data: default=%default')


    (options, args) = parser.parse_args()
    files = args
    n_files = len(files)

    use_most_similar = options.similar
    use_least_similar = options.different
    use_balanced = options.balanced
    use_unbalanced = options.unbalanced
    percentile = int(options.percentile)
    twitter = options.twitter

    output = options.prefix

    rows = ['train', 'CC', 'PCC', 'ACC_internal', 'PCC_platt2']

    datasets = ['cshift', 'f1', 'acc', 'cal']

    values = {}
    for row in rows:
        for d in datasets:
            values[row + '_' + d] = []

    dfs = []

    for d_i, d in enumerate(datasets):
        print(d)
        if d != 'cshift':
            if twitter:
                current_files = [re.sub('cshift_100000_', '', f) for f in files]
                current_files = [re.sub('cshift', '_0', f) for f in current_files]
            else:
                current_files = [re.sub('cshift_', '', f) for f in files]
                current_files = [re.sub('acc_', 'f1_', f) for f in current_files]
        else:
            current_files = [f for f in files]
        if d == 'acc':
            current_files = [re.sub('f1_', 'acc_', f) for f in current_files]
        elif d == 'cal':
            current_files = [re.sub('f1_', 'calibration_', f) for f in current_files]

        df = None
        mae_values = None
        train_estimates = []
        train_maes = []
        for f_i, f in enumerate(current_files):
            print(f)
            n_files += 1
            df_f = fh.read_csv_to_df(f)
            n_rows, n_cols = df_f.shape
            train_estimates.append(df_f.loc['train', 'estimate'])

            n_train = int(df_f.loc['train', 'N'])
            for row in rows:
                values[row + '_' + d].append(df_f.loc[row, 'MAE'])

        print("%d files" % len(files))

    """
    df = pd.DataFrame(mae_values, index=df.index)

    most_similar = train_maes < np.mean(train_maes)
    least_similar = train_maes > np.mean(train_maes)
    train_unalancedness = np.abs(np.array(train_estimates) - 0.5)
    most_balanced = train_unalancedness < np.mean(train_unalancedness)
    least_balanced = train_unalancedness > np.mean(train_unalancedness)

    selector = np.array(np.ones(len(most_similar)), dtype=bool)
    if use_most_similar:
        selector *= most_similar
    if use_least_similar:
        selector *= least_similar
    if use_balanced:
        selector *= most_balanced
    if use_unbalanced:
        selector *= least_balanced

    df = pd.DataFrame(df.values[:, selector], index=df.index)
    print(df.mean(axis=1))
    print(df.std(axis=1))
    """

    for row, numbers in values.items():
        print(row, len(numbers), np.mean(numbers))

    to_plot = ['train_f1', 'CC_f1', 'PCC_acc', 'PCC_f1', 'PCC_platt2_f1', 'PCC_cshift', 'ACC_internal_f1', 'PCC_cal']
    #names = ['Train', 'PCC(cal)', 'PCC(F1)', 'PCC(acc)', 'CC', 'Platt', 'Reweighting', 'ACC']
    names = ['Train', 'CC', 'PCC(acc)', 'PCC(F1)', 'Platt', 'Reweighting', 'ACC', 'PCC(cal)']
    names = ['Train', 'CC', r'PCC$^{\mathrm{acc}}$', r'PCC$^{\mathrm{F}_1}$', 'Platt', 'Reweighting', 'ACC', 'PCC$^{\mathrm{cal}}$']

    rows = [values[r] for r in to_plot]
    rows = np.vstack(rows)

    train_maes = values['train_f1']
    most_similar = train_maes <= np.percentile(train_maes, q=percentile)
    least_similar = train_maes > np.percentile(train_maes, q=100-percentile)
    train_unalancedness = np.abs(np.array(train_estimates) - 0.5)
    most_balanced = train_unalancedness <= np.percentile(train_unalancedness, q=percentile)
    least_balanced = train_unalancedness > np.percentile(train_unalancedness, q=100-percentile)

    selector = np.array(np.ones(len(most_similar)), dtype=bool)
    if use_most_similar:
        selector *= most_similar
    if use_least_similar:
        selector *= least_similar
    if use_balanced:
        selector *= most_balanced
    if use_unbalanced:
        selector *= least_balanced

    print(rows.shape)
    rows = rows[:, selector]
    print(rows.shape)

    means = np.mean(rows, axis=1)

    values_df = pd.DataFrame(rows, index=names)
    values_df.to_csv(output + '.csv')

    means_df = pd.DataFrame(means, index=names)
    print(means_df)

    y = list(range(len(names)))
    y.reverse()

    fig, ax = plt.subplots()
    ax.barh(y, means, alpha=0.5, facecolor='blue')
    for y_i, y_val in enumerate(y):
        #vals = values[to_plot[y_i]]
        vals = values_df.loc[names[y_i]]
        ax.scatter(vals, np.ones_like(vals) * y_val, s=10, facecolor='k', alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, 0.39)
    plt.savefig(output + '.pdf', bbox_inches='tight')

    #if output is not None:
    #    df.to_csv(output + '.csv')
    #    df.mean(axis=1).to_csv(output + '_mean.csv')

    """
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, len(rows)))

    fig, ax = plt.subplots()
    for r_i, row in enumerate(rows):
        means = []
        groups = list(values[row].keys())
        groups.sort()
        for group in groups:
            points = values[row][group]
            n_points = len(points)
            ax.scatter(np.ones(n_points)*group + r_i*8, points, color=colors[r_i], s=5, alpha=0.5)
            means.append(np.mean(points))
        if row == 'train':
            ax.plot(groups, means, linestyle='dashed', color=colors[r_i], label=row, alpha=0.5)
        else:
            ax.plot(groups, means, color=colors[r_i], label=row, alpha=0.5)
    ax.legend()
    if output is not None:
        plt.savefig(output + '.pdf', bbox_inches='tight')
    """

if __name__ == '__main__':
    main()
