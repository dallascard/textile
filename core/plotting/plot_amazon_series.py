import os
import sys
import glob
from optparse import OptionParser

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    input_dir = 'projects/amazon/sports5/models/'
    test_vals = []
    train_vals = []
    pcc_f1_vals = []
    acc_f1_vals = []
    cal_f1_vals = []
    platt_vals = []
    acc_vals = []
    cshift_vals = []

    years = [2009, 2010, 2011, 2012, 2013, 2014]
    for year in ['2009', '2010', '2011', '2012', '2013', '2014']:
        acc_file = os.path.join(input_dir, 'all_helpfulness_l1_acc_2000_0_sampled_' + year + '-' + year + '_0', 'results.csv')
        cal_file = os.path.join(input_dir, 'all_helpfulness_l1_calibration_2000_0_sampled_' + year + '-' + year + '_0', 'results.csv')
        f1_file = os.path.join(input_dir, 'all_helpfulness_l1_f1_2000_0_sampled_' + year + '-' + year + '_0', 'results.csv')
        #cshift_file = os.path.join(input_dir, 'all_helpfulness_l1_f1_2000_0_sampled_cshift_' + year + '-' + year + '_0', 'results.csv')

        acc_df = pd.read_csv(acc_file, index_col=0, header=0)
        cal_df = pd.read_csv(cal_file, index_col=0, header=0)
        f1_df = pd.read_csv(f1_file, index_col=0, header=0)
        #cshift_df = pd.read_csv(cshift_file, index_col=0, header=0)

        test_vals.append(f1_df['estimate'].loc['test'])
        train_vals.append(f1_df['estimate'].loc['train'])

    fig, ax = plt.subplots()
    ax.plot(years, test_vals, label='Test')
    ax.plot(years, train_vals, label='Train')
    ax.legend()
    plt.savefig('test.pdf')


if __name__ == '__main__':
    main()
