from optparse import OptionParser

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
    n_files = 0
    for f in files:
        n_files += 1
        df_f = fh.read_csv_to_df(f)
        if df is None:
            df = df_f
        else:
            df += df_f
    df = df / float(n_files)
    print(df)


if __name__ == '__main__':
    main()
