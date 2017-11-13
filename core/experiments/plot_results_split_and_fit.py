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

    rows = ['CC', 'PCC', 'ACC', 'MS', 'PCC_platt2', 'PCC_DL']
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

        n_train = df_f.loc['train', 'N']
        if n_train not in values['CC']:
            for row in rows:
                values[row][n_train] = []
        for row in rows:
            values[row][n_train].append(df_f.loc[row, 'MAE'])

    df = pd.DataFrame(mae_values, index=df.index)
    print(df.mean(axis=1))
    print(df.var(axis=1))

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
            ax.scatter(np.ones(n_points)*group, points, color=colors[r_i])
            means.append(points)
        ax.plot(groups, means, color=colors[r_i])
    plt.savefig('test.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
