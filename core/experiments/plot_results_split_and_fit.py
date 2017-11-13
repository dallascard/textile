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
    parser.add_option('--output', dest='output', default=None,
                      help='Output filename (optional): default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()
    files = args
    n_files = len(files)

    output = options.output

    rows = ['train', 'CC', 'PCC', 'ACC_internal', 'MS_internal', 'PCC_platt2', 'PCC_DL']
    values = {}
    for row in rows:
        values[row] = {}

    df = None
    mae_values = None
    for f_i, f in enumerate(files):
        print(f)
        n_files += 1
        df_f = fh.read_csv_to_df(f)
        n_rows, n_cols = df_f.shape
        if mae_values is None:
            df = df_f
            mae_values = np.zeros([n_rows, n_files-1])
        mae_values[:, f_i] = df_f['MAE'].values

        n_train = int(df_f.loc['train', 'N'])
        if n_train not in values['CC']:
            for row in rows:
                values[row][n_train] = []
        for row in rows:
            values[row][n_train].append(df_f.loc[row, 'MAE'])

    df = pd.DataFrame(mae_values, index=df.index)
    print(df.mean(axis=1))
    print(df.var(axis=1))

    train_mean = df.loc['train'].mean()
    most_similar = df.loc['train'].values < train_mean
    most_different = df.loc['train'].values > train_mean

    print("Most similar")
    print(df.values[:, most_similar].mean(axis=1))

    print("Most different")
    print(df.values[:, most_different].mean(axis=1))


    if output is not None:
        df.to_csv(output)

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
            ax.scatter(np.ones(n_points)*group, points, color=colors[r_i], s=5, alpha=0.5)
            means.append(np.mean(points))
        if row == 'train':
            ax.plot(groups, means, linestyle='dashed', color=colors[r_i], label=row, alpha=0.5)
        else:
            ax.plot(groups, means, color=colors[r_i], label=row, alpha=0.5)
    ax.legend()
    plt.savefig('test.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
