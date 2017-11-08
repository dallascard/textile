from optparse import OptionParser

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

    df = None
    values = None
    n_files = 0
    for f in files:
        print(f)
        n_files += 1
        df_f = fh.read_csv_to_df(f)
        if values is None:
            df = df_f
            values = df['MAE'].values.copy()
        else:
            values += df_f['MAE'].values
    print(values)
    print(n_files)
    values = values / float(n_files)
    df = pd.DataFrame(values, columns=['MAE'], index=df.index)
    print(df)


if __name__ == '__main__':
    main()
