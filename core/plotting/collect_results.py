import os
import re
from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog csv_results_files"
    parser = OptionParser(usage=usage)
    parser.add_option('--prefix', dest='prefix', default='test',
                      help='Output prefix (optional): default=%default')

    (options, args) = parser.parse_args()
    files = args
    n_files = len(files)

    output = options.prefix

    df = None

    for f_i, f in enumerate(files):
        print(f)
        n_files += 1
        df_f = fh.read_csv_to_df(f)
        if df is None:
            df = pd.DataFrame(df_f['MAE'], columns=[f], index=df_f.index)
        else:
            df[f] = df_f['MAE']

    print("%d files" % len(files))
    df.to_csv(output + '.csv')


if __name__ == '__main__':
    main()
