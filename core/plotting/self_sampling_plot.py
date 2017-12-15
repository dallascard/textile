import os
import glob

from optparse import OptionParser

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from ..util import file_handling as fh


# Plotting function for core.experiments.self_sampling

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    n_trains = [100, 200, 500, 1000, 2000, 5000, 10000]

    srs = []
    pcc = []

    for n in n_trains:
        lines = fh.read_text('self_sampling_' + str(n) + '.txt')
        srs.append(float(lines[0].strip()))
        pcc.append(float(lines[1].strip()))

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(n_trains, srs, label='SRS')
    ax.plot(n_trains, pcc, label='PCC')
    ax.set_xlabel('Amount of labeled data (L)')
    ax.set_ylabel('Mean AE')
    ax.legend()
    plt.savefig('self_sampling.pdf', bbox_inches='tight')


if __name__ == '__main__':
    main()
