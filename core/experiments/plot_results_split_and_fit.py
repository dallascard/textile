from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog csv_results_files"
    parser = OptionParser(usage=usage)
    parser.add_option('--prefix', dest='prefix', default=None,
                      help='Output prefix (optional): default=%default')
    parser.add_option('--similar', action="store_true", dest="similar", default=False,
                      help='Only use the most similar examples: default=%default')
    parser.add_option('--different', action="store_true", dest="different", default=False,
                      help='Only use the most different examples: default=%default')
    parser.add_option('--balanced', action="store_true", dest="balanced", default=False,
                      help='Only use the most balanced examples: default=%default')
    parser.add_option('--unbalanced', action="store_true", dest="unbalanced", default=False,
                      help='Only use the most unbalanced examples: default=%default')


    (options, args) = parser.parse_args()
    files = args
    n_files = len(files)

    use_most_similar = options.similar
    use_least_similar = options.different
    use_balanced = options.balanced
    use_unbalanced = options.unbalanced

    output = options.prefix

    rows = ['train', 'CC', 'PCC', 'ACC_internal', 'MS_internal', 'PCC_platt2']
    values = {}
    for row in rows:
        values[row] = {}

    df = None
    mae_values = None
    train_estimates = []
    train_maes = []
    for f_i, f in enumerate(files):
        print(f)
        n_files += 1
        df_f = fh.read_csv_to_df(f)
        n_rows, n_cols = df_f.shape
        if mae_values is None:
            df = df_f
            mae_values = np.zeros([n_rows, n_files-1])
        mae_values[:, f_i] = df_f['MAE'].values

        train_estimates.append(df_f.loc['train', 'estimate'])
        train_maes.append(df_f.loc['train', 'MAE'])

        n_train = int(df_f.loc['train', 'N'])
        if n_train not in values['CC']:
            for row in rows:
                values[row][n_train] = []
        for row in rows:
            values[row][n_train].append(df_f.loc[row, 'MAE'])
    print("%d files" % len(files))

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

    if output is not None:
        df.mean(axis=1).to_csv(output + '.csv')

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


if __name__ == '__main__':
    main()
