from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog csv_results_files"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()
    files = args
    n_files = len(files)

    df = None
    values = None
    for f_i, f in enumerate(files):
        print(f)
        n_files += 1
        df_f = fh.read_csv_to_df(f)
        if values is None:
            n_rows, n_cols = df_f.shape
            values = np.zeros([n_rows, n_files])
        values[:, f_i] = df_f['MAE'].values

    values = values / float(n_files)
    df = pd.DataFrame(values, index=df.index)
    print(df.mean(axis=1))
    print(df.var(axis=1))
    print(df)


if __name__ == '__main__':
    main()
