import os
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from ..util import file_handling as fh


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--cshift', dest='cshift', default=None,
                      help='cshift [None|classify]: default=%default')
    parser.add_option('-t', dest='train_prop', default=0.9,
                      help='Train prop: default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=100,
                      help='Number of calibration instances: default=%default')
    parser.add_option('--base', dest='base', default='mfc',
                      help='base [mfc|amazon]: default=%default')
    parser.add_option('--subset', dest='subset', default='*',
                      help='Subset of base (e.g. immigration: default=%default')
    parser.add_option('--label', dest='label', default='*',
                      help='Label (e.g. Economic: default=%default')
    #parser.add_option('--partition', dest='partition', default='*',
    #                  help='Partition for mfc (e.g. pre: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='model type [LR|MLP]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='objective [f1|calibration]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')


    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    objective = options.objective
    cshift = options.cshift
    train_prop = str(float(options.train_prop))
    n_calib = str(int(options.n_calib))
    base = options.base
    subset = options.subset
    label = options.label
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    basename = '*_' + label + '_*_' + model_type + '_' + penalty
    if model_type == 'MLP':
        basename += '_' + dh
    basename += '_' + train_prop + '_' + n_calib + '_' + objective
    if model_type == 'MLP':
        basename += '_r?'
    if cshift is not None:
        basename += '_cshift'
    if base == 'mfc':
        basename += '_???_????_?'
    elif base == 'amazon':
        basename += '_????_?'

    print(basename)
    files = glob(os.path.join('projects', base, subset, 'models', basename, 'results.csv'))
    files.sort()
    n_files = len(files)

    print(files[0])
    results = fh.read_csv_to_df(files[0])
    df = pd.DataFrame(results[['estimate', 'RMSE', 'contains_test']].copy())

    venn_outside_errors = []
    n_outside = 0
    calib_rmses = []
    calib_widths = []
    PCC_cal_rmses = []
    PCC_cal_overestimates = []
    PCC_nontrain_rmses = []
    PCC_nontrain_overestimates = []
    venn_rmses = []
    venn_widths = []
    cv_cals = []
    cv_f1s = []
    calibration_cals = []
    calibration_f1s = []

    target_prop = results.loc['target', 'estimate']
    venn_av_lower = results.loc['Venn_averaged', '95lcl']
    venn_av_upper = results.loc['Venn_averaged', '95ucl']
    venn_inside = results.loc['Venn_averaged', 'contains_test']

    if venn_inside == 0:
        venn_outside_errors.append(max(venn_av_lower - target_prop, target_prop - venn_av_upper))
        n_outside += 1

    calib_rmses.append(results.loc['calibration', 'RMSE'])
    calib_widths.append(results.loc['calibration', '95ucl'] - results.loc['calibration', '95lcl'])
    PCC_cal_rmses.append(results.loc['PCC_cal', 'RMSE'])
    PCC_nontrain_rmses.append(results.loc['PCC_nontrain', 'RMSE'])
    PCC_cal_overestimates.append(results.loc['PCC_cal', 'estimate'] - results.loc['calibration', 'estimate'])
    PCC_nontrain_overestimates.append(results.loc['PCC_nontrain', 'estimate'] - results.loc['target', 'estimate'])
    venn_rmses.append(results.loc['Venn', 'RMSE'])
    venn_widths.append(results.loc['Venn_averaged', '95ucl'] - results.loc['Venn_averaged', '95lcl'])

    file_dir, _ = os.path.split(files[0])
    accuracy_file = os.path.join(file_dir, 'accuracy.csv')
    accuracy_df = fh.read_csv_to_df(accuracy_file)
    cv_cals.append(accuracy_df.loc['cross_val', 'calibration'])
    cv_f1s.append(accuracy_df.loc['cross_val', 'f1'])
    calibration_cals.append(accuracy_df.loc['calibration', 'calibration'])
    calibration_f1s.append(accuracy_df.loc['calibration', 'f1'])

    for f in files[1:]:
        print(f)
        results = fh.read_csv_to_df(f)
        df += results[['estimate', 'RMSE', 'contains_test']]

        target_prop = results.loc['target', 'estimate']
        venn_av_lower = results.loc['Venn_averaged', '95lcl']
        venn_av_upper = results.loc['Venn_averaged', '95ucl']
        venn_inside = results.loc['Venn_averaged', 'contains_test']

        if venn_inside == 0:
            venn_outside_errors.append(max(venn_av_lower - target_prop, target_prop - venn_av_upper))
            n_outside += 1

        calib_rmses.append(results.loc['calibration', 'RMSE'])
        calib_widths.append(results.loc['calibration', '95ucl'] - results.loc['calibration', '95lcl'])
        PCC_cal_rmses.append(results.loc['PCC_cal', 'RMSE'])
        PCC_nontrain_rmses.append(results.loc['PCC_nontrain', 'RMSE'])
        PCC_cal_overestimates.append(results.loc['PCC_cal', 'estimate'] - results.loc['calibration', 'estimate'])
        PCC_nontrain_overestimates.append(results.loc['PCC_nontrain', 'estimate'] - results.loc['target', 'estimate'])
        venn_rmses.append(results.loc['Venn', 'RMSE'])
        venn_widths.append(results.loc['Venn', '95ucl'] - results.loc['Venn', '95lcl'])

        file_dir, _ = os.path.split(f)
        accuracy_file = os.path.join(file_dir, 'accuracy.csv')
        accuracy_df = fh.read_csv_to_df(accuracy_file)
        cv_cals.append(accuracy_df.loc['cross_val', 'calibration'])
        cv_f1s.append(accuracy_df.loc['cross_val', 'f1'])
        calibration_cals.append(accuracy_df.loc['calibration', 'calibration'])
        calibration_f1s.append(accuracy_df.loc['calibration', 'f1'])

    df = df / float(n_files)

    print(df)
    print("n_outside: %d" % n_outside)
    print("mean venn outside error = %0.6f" % np.mean(venn_outside_errors))
    print(" max venn outside error = %0.6f" % np.max(venn_outside_errors))
    print("mean calib width = %0.4f" % np.mean(calib_widths))
    print("mean venn widths = %0.6f" % np.mean(venn_widths))

    corr, p_val = pearsonr(PCC_nontrain_rmses, cv_cals)
    print("PCC correlation (with cv_cal) = %0.4f" % corr)
    corr, p_val = pearsonr(PCC_nontrain_rmses, cv_f1s)
    print("PCC correlation (with cv_f1s) = %0.4f" % corr)

    corr, p_val = pearsonr(PCC_nontrain_rmses, calibration_cals)
    print("PCC correlation (with calib calibration) = %0.4f" % corr)
    corr, p_val = pearsonr(PCC_nontrain_rmses, calibration_f1s)
    print("PCC correlation (with calib f1s) = %0.4f" % corr)

    corr, p_val = pearsonr(PCC_nontrain_rmses, PCC_cal_rmses)
    print("PCC correlation (with PCC_cal) = %0.4f" % corr)

    corr, p_val = pearsonr(PCC_nontrain_overestimates, PCC_cal_overestimates)
    print("PCC correlation (with PCC_cal) = %0.4f" % corr)

    corr, p_val = pearsonr(venn_rmses, PCC_cal_rmses)
    print("Venn correlation (with PCC_cal) = %0.4f" % corr)

    #plt.scatter(PCC_cal_rmses, PCC_nontrain_rmses)
    plt.scatter(PCC_cal_overestimates, PCC_nontrain_overestimates)
    plt.plot((np.min(PCC_cal_overestimates), np.max(PCC_cal_overestimates)), (np.min(PCC_nontrain_overestimates), np.max(PCC_nontrain_overestimates)))
    #plt.plot((np.min(PCC_cal_rmses), np.max(PCC_cal_rmses)), (np.min(PCC_nontrain_rmses), np.max(PCC_nontrain_rmses)))
    #plt.xlabel('PCC_cal_rmse')
    #plt.ylabel('PCC_nontrain_rmse')
    #plt.savefig('test.pdf')
    plt.xlabel('PCC_cal_overestimate')
    plt.ylabel('PCC_nontrain_overestimate')
    plt.show()

    # repeat for accuracy / f1
    files = glob(os.path.join('projects', base, subset, 'models', basename, 'accuracy.csv'))
    files.sort()
    n_files = len(files)

    #print(files[0])
    results = fh.read_csv_to_df(files[0])
    df = results.copy()

    for f in files[1:]:
        #print(f)
        results = fh.read_csv_to_df(f)
        df += results

    df = df / float(n_files)
    print(df)


if __name__ == '__main__':
    main()
