import os
from glob import glob
from optparse import OptionParser

import pandas as pd

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    files = glob(os.path.join('projects', 'mfc', 'samesex', 'models', '*_year_group_LR_l2_0.9_0.1_f1_*', 'results.csv'))
    files.sort()
    n_files = len(files)

    results = fh.read_csv_to_df(files[0])
    df = pd.DataFrame(results[['estimate', 'contains_test']].copy())

    for f in files[1:]:
        results = fh.read_csv_to_df(f)
        df += results[['estimate', 'contains_test']]

    df = df / float(n_files)

    print(df)


if __name__ == '__main__':
    main()
