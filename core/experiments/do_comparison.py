from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

from ..util import file_handling as fh


def main():
    usage = "%prog csv1 csv2 r1 r2"
    parser = OptionParser(usage=usage)
    #parser.add_option('--prefix', dest='prefix', default=None,
    #                  help='Output prefix (optional): default=%default')
    #parser.add_option('--similar', action="store_true", dest="similar", default=False,
    #                  help='Only use the most similar examples: default=%default')
    #parser.add_option('--different', action="store_true", dest="different", default=False,
    #                  help='Only use the most different examples: default=%default')
    #parser.add_option('--balanced', action="store_true", dest="balanced", default=False,
    #                  help='Only use the most balanced examples: default=%default')
    #parser.add_option('--unbalanced', action="store_true", dest="unbalanced", default=False,
    #                  help='Only use the most unbalanced examples: default=%default')


    (options, args) = parser.parse_args()

    csv1 = args[0]
    csv2 = args[1]
    r1 = int(args[2])
    r2 = int(args[3])

    df1 = pd.read_csv(csv1, index_col=0, header=0)
    df2 = pd.read_csv(csv2, index_col=0, header=0)

    print("Comparing %s to %s" % (df1.index[r1], df2.index[r2]))
    values1 = df1.iloc[r1].values
    values2 = df2.iloc[r2].values

    temp1 = []
    temp2 = []
    n_values = len(values1)
    for i in range(n_values//5):
        temp1.append(np.mean(values1[i*5:(i+1)*5]))
        temp2.append(np.mean(values2[i*5:(i+1)*5]))

    values1 = np.array(temp1)
    values2 = np.array(temp2)

    print(np.mean(values1))
    print(np.mean(values2))
    print(values1 - values2)
    print(np.mean(values1 - values2))
    print(ttest_rel(values1, values2))

    values = np.r_[values1, values2]
    labels = np.repeat((0, 1), len(values1))

    diffs = []
    for i in range(1000):
        np.random.shuffle(labels)
        diff = values[labels == 0] - values[labels == 1]
        diffs.append(diff)
    print(np.percentile(diffs, 5), np.percentile(diffs, 50), np.median(diffs), np.percentile(diffs, 95))

if __name__ == '__main__':
    main()
