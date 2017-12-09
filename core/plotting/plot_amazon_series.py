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
    target_vals = []
    target_stds = []
    train_vals = []
    pcc_f1_vals = []
    pcc_acc_vals = []
    pcc_cal_vals = []
    platt_vals = []
    #acc_vals = []
    cshift_vals = []

    years = [2009, 2010, 2011, 2012, 2013, 2014]
    for year in ['2009', '2010', '2011', '2012', '2013', '2014']:
        acc_file = os.path.join(input_dir, 'all_helpfulness_l1_acc_2000_0_sampled_' + year + '-' + year + '_0', 'results.csv')
        cal_file = os.path.join(input_dir, 'all_helpfulness_l1_calibration_2000_0_sampled_' + year + '-' + year + '_0', 'results.csv')
        f1_file = os.path.join(input_dir, 'all_helpfulness_l1_f1_2000_0_sampled_' + year + '-' + year + '_0', 'results.csv')
        cshift_file = os.path.join(input_dir, 'all_helpfulness_l1_f1_2000_0_sampled_cshift_' + year + '-' + year + '_0', 'results.csv')

        acc_df = pd.read_csv(acc_file, index_col=0, header=0)
        cal_df = pd.read_csv(cal_file, index_col=0, header=0)
        f1_df = pd.read_csv(f1_file, index_col=0, header=0)
        cshift_df = pd.read_csv(cshift_file, index_col=0, header=0)

        target_vals.append(f1_df['estimate'].loc['target'])
        target_p = f1_df['estimate'].loc['target']
        target_n = f1_df['N'].loc['target']
        target_stds.append(np.sqrt(target_p * (1- target_p) / target_n))
        train_vals.append(f1_df['estimate'].loc['train'])
        pcc_f1_vals.append(f1_df['estimate'].loc['PCC'])
        platt_vals.append(f1_df['estimate'].loc['PCC_platt2'])
        #acc_vals.append(f1_df['estimate'].loc['ACC_internal'])
        pcc_acc_vals.append(acc_df['estimate'].loc['PCC'])
        pcc_cal_vals.append(cal_df['estimate'].loc['PCC'])
        cshift_vals.append(cshift_df['estimate'].loc['PCC'])

    fig, ax = plt.subplots()
    ax.plot(years, target_vals, label='Target')
    ax.fill_between(years, np.array(target_vals) - 2 * np.array(target_stds), np.array(target_vals) + 2 * np.array(target_stds), color='k', alpha=0.1)
    ax.scatter(years, train_vals, label='Train sample mean')
    ax.plot(years, train_vals, linestyle='--', alpha=0.5, label='Train sample mean')
    ax.plot(years, pcc_cal_vals, label='PCC(cal)')
    ax.plot(years, pcc_acc_vals, label='PCC(acc)')
    ax.plot(years, pcc_f1_vals, label='PCC(F1)')
    ax.plot(years, platt_vals, label='Platt')
    ax.plot(years, cshift_vals, label='Reweighting')
    ax.set_ylim(0.5, 1.0)
    #ax.plot(years, acc_vals, label='ACC')
    ax.legend()
    plt.savefig('test.pdf')


if __name__ == '__main__':
    main()
