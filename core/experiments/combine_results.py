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
    parser.add_option('--n_train', dest='n_train', default=100,
                      help='Train prop: default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=100,
                      help='Number of calibration instances: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--base', dest='base', default='mfc',
                      help='base [mfc|amazon]: default=%default')
    parser.add_option('--subset', dest='subset', default='*',
                      help='Subset of base (e.g. immigration: default=%default')
    parser.add_option('--label', dest='label', default='*',
                      help='Label (e.g. Economic: default=%default')
    parser.add_option('--partition', dest='partition', default=None,
                      help='Partition for mfc (e.g. pre: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='model type [LR|MLP]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
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
    n_train = str(int(options.n_train))
    n_calib = str(int(options.n_calib))
    sample_labels = options.sample
    base = options.base
    subset = options.subset
    label = options.label
    partition = options.partition
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    # basic LR f1: combining subset, label, repetitions, and pre/post date
    #basename = '*_' + model_type
    basename = '*_' + label + '_*_' + model_type + '_' + penalty
    if model_type == 'MLP':
        basename += '_' + dh
    basename += '_' + n_train + '_' + n_calib + '_' + objective
    if model_type == 'MLP':
        basename += '_r?'
    if cshift is not None:
        basename += '_cshift'
    if sample_labels:
        basename += '_sampled'
    if base == 'mfc':
        if partition is None:
            basename += '_???_????_?'
        else:
            basename += '_' + partition + '_?'
    elif base == 'amazon':
        if partition is None:
            basename += '_????_?'
        else:
            basename += '_' + partition + '_?'

    print(basename)
    files = glob(os.path.join('projects', base, subset, 'models', basename, 'results.csv'))
    files.sort()
    n_files = len(files)

    print(files[0])
    results = fh.read_csv_to_df(files[0])
    df = pd.DataFrame(columns=['estimate', 'MAE', 'MSE', 'contains_test', 'max_MAE'])
    #df = pd.DataFrame(results[['estimate', 'RMSE', 'contains_test']].copy())
    target_estimate = results.loc['target', 'estimate']
    for loc in results.index:
        df.loc[loc, 'estimate'] = results.loc[loc, 'estimate']
        df.loc[loc, 'MAE'] = results.loc[loc, 'RMSE']
        df.loc[loc, 'contains_test'] = results.loc[loc, 'contains_test']
        df.loc[loc, 'MSE'] = (results.loc[loc, 'estimate'] - target_estimate) ** 2
        df.loc[loc, 'max_MAE'] = results.loc[loc, 'RMSE']

    venn_outside_errors = []
    n_outside = 0
    calib_rmses = []
    calib_widths = []
    calib_widths_n_annotations = []
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
    adj_errors = []
    venn_calib_in_range_vals = []
    venn_levels_vals = []

    target_prop = results.loc['target', 'estimate']
    venn_av_lower = results.loc['Venn_averaged', '95lcl']
    venn_av_upper = results.loc['Venn_averaged', '95ucl']
    venn_inside = results.loc['Venn_averaged', 'contains_test']

    if venn_inside == 0:
        venn_outside_errors.append(max(venn_av_lower - target_prop, target_prop - venn_av_upper))
        n_outside += 1

    calib_rmses.append(results.loc['calibration', 'RMSE'])
    calib_widths.append(results.loc['calibration', '95ucl'] - results.loc['calibration', '95lcl'])
    calib_widths_n_annotations.append(results.loc['calibration_n_annotations', '95ucl'] - results.loc['calibration_n_annotations', '95lcl'])
    PCC_cal_rmses.append(results.loc['PCC_cal', 'RMSE'])
    PCC_nontrain_rmses.append(results.loc['PCC_nontrain', 'RMSE'])
    PCC_cal_overestimates.append(results.loc['PCC_cal', 'estimate'] - results.loc['calibration', 'estimate'])
    PCC_nontrain_overestimates.append(results.loc['PCC_nontrain', 'estimate'] - results.loc['target', 'estimate'])
    venn_rmses.append(results.loc['Venn', 'RMSE'])
    venn_widths.append(results.loc['Venn_averaged', '95ucl'] - results.loc['Venn_averaged', '95lcl'])
    cal_error_estimate = results.loc['PCC_cal', 'estimate'] - results.loc['calibration', 'estimate']
    adj_errors.append(results.loc['PCC_nontrain', 'estimate'] - cal_error_estimate - results.loc['target', 'estimate'])

    file_dir, _ = os.path.split(files[0])
    accuracy_file = os.path.join(file_dir, 'accuracy.csv')
    accuracy_df = fh.read_csv_to_df(accuracy_file)
    cv_cals.append(accuracy_df.loc['cross_val', 'calibration'])
    cv_f1s.append(accuracy_df.loc['cross_val', 'f1'])
    #calibration_cals.append(accuracy_df.loc['calibration', 'calibration'])
    #calibration_f1s.append(accuracy_df.loc['calibration', 'f1'])

    venn_range_file = os.path.join(file_dir, 'venn_calib_props_in_range.csv')
    venn_calib_in_range_list = [float(f) for f in fh.read_text(venn_range_file)]
    venn_calib_in_range_vals.append(np.mean(venn_calib_in_range_list))

    venn_levels_file = os.path.join(file_dir, 'list_of_n_levels.csv')
    venn_levels_list = [float(f) for f in fh.read_text(venn_levels_file)]
    venn_levels_vals.append(np.mean(venn_levels_list))

    for f in files[1:]:
        print(f)
        results = fh.read_csv_to_df(f)
        #df += results[['estimate', 'RMSE', 'contains_test']]

        target_estimate = results.loc['target', 'estimate']
        for loc in results.index:
            df.loc[loc, 'estimate'] += results.loc[loc, 'estimate']
            df.loc[loc, 'MAE'] += results.loc[loc, 'RMSE']
            df.loc[loc, 'contains_test'] += results.loc[loc, 'contains_test']
            df.loc[loc, 'MSE'] += (results.loc[loc, 'estimate'] - target_estimate) ** 2
            df.loc[loc, 'max_MAE'] = max(df.loc[loc, 'max_MAE'], results.loc[loc, 'RMSE'])

        target_prop = results.loc['target', 'estimate']
        venn_av_lower = results.loc['Venn_averaged', '95lcl']
        venn_av_upper = results.loc['Venn_averaged', '95ucl']
        venn_inside = results.loc['Venn_averaged', 'contains_test']

        if venn_inside == 0:
            venn_outside_errors.append(max(venn_av_lower - target_prop, target_prop - venn_av_upper))
            n_outside += 1

        calib_rmses.append(results.loc['calibration', 'RMSE'])
        calib_widths.append(results.loc['calibration', '95ucl'] - results.loc['calibration', '95lcl'])
        calib_widths_n_annotations.append(results.loc['calibration_n_annotations', '95ucl'] - results.loc['calibration_n_annotations', '95lcl'])
        PCC_cal_rmses.append(results.loc['PCC_cal', 'RMSE'])
        PCC_nontrain_rmses.append(results.loc['PCC_nontrain', 'RMSE'])
        PCC_cal_overestimates.append(results.loc['PCC_cal', 'estimate'] - results.loc['calibration', 'estimate'])
        PCC_nontrain_overestimates.append(results.loc['PCC_nontrain', 'estimate'] - results.loc['target', 'estimate'])
        venn_rmses.append(results.loc['Venn', 'RMSE'])
        venn_widths.append(results.loc['Venn', '95ucl'] - results.loc['Venn', '95lcl'])
        cal_error_estimate = results.loc['PCC_cal', 'estimate'] - results.loc['calibration', 'estimate']
        adj_errors.append(np.abs(results.loc['PCC_nontrain', 'estimate'] - cal_error_estimate - results.loc['target', 'estimate']))

        file_dir, _ = os.path.split(f)
        accuracy_file = os.path.join(file_dir, 'accuracy.csv')
        accuracy_df = fh.read_csv_to_df(accuracy_file)
        cv_cals.append(accuracy_df.loc['cross_val', 'calibration'])
        cv_f1s.append(accuracy_df.loc['cross_val', 'f1'])
        #calibration_cals.append(accuracy_df.loc['calibration', 'calibration'])
        #calibration_f1s.append(accuracy_df.loc['calibration', 'f1'])

        venn_range_file = os.path.join(file_dir, 'venn_calib_props_in_range.csv')
        venn_calib_in_range_list = [float(f) for f in fh.read_text(venn_range_file)]
        venn_calib_in_range_vals.append(np.mean(venn_calib_in_range_list))

        venn_levels_file = os.path.join(file_dir, 'list_of_n_levels.csv')
        venn_levels_list = [float(f) for f in fh.read_text(venn_levels_file)]
        venn_levels_vals.append(np.mean(venn_levels_list))

    #df = df / float(n_files)
    df['estimate'] = df['estimate'] / float(n_files)
    df['MAE'] = df['MAE'] / float(n_files)
    df['MSE'] = df['MSE'] / float(n_files)
    df['contains_test'] = df['contains_test'] / float(n_files)



    print(df)
    print("Mean adjusted error rmse = %0.5f" % np.mean(adj_errors))
    print("n_outside: %d" % n_outside)
    print("mean venn outside error = %0.6f" % np.mean(venn_outside_errors))
    print(" max venn outside error = %0.6f" % np.max(venn_outside_errors))
    print("mean calib (n_items) width = %0.4f" % np.mean(calib_widths))
    print("mean calib (n_annot) width = %0.4f" % np.mean(calib_widths_n_annotations))
    print("mean venn widths = %0.6f" % np.mean(venn_widths))

    corr, p_val = pearsonr(PCC_nontrain_rmses, cv_cals)
    print("PCC correlation (with cv_cal) = %0.4f" % corr)
    corr, p_val = pearsonr(PCC_nontrain_rmses, cv_f1s)
    print("PCC correlation (with cv_f1s) = %0.4f" % corr)

    corr, p_val = pearsonr(PCC_nontrain_rmses, PCC_cal_rmses)
    print("PCC correlation (with PCC_cal) = %0.4f" % corr)

    corr, p_val = pearsonr(PCC_nontrain_overestimates, PCC_cal_overestimates)
    print("PCC correlation (with PCC_cal) = %0.4f" % corr)

    corr, p_val = pearsonr(venn_rmses, PCC_cal_rmses)
    print("Venn correlation (with PCC_cal) = %0.4f" % corr)

    corr, p_val = pearsonr(venn_rmses, venn_calib_in_range_vals)
    print("Venn correlation (with venn calib in range) = %0.4f" % corr)

    corr, p_val = pearsonr(venn_rmses, venn_levels_vals)
    print("Venn correlation (with venn levels) = %0.4f" % corr)

    #plt.scatter(PCC_cal_rmses, PCC_nontrain_rmses)
    #plt.scatter(PCC_cal_overestimates, PCC_nontrain_overestimates)
    #plt.plot((np.min(PCC_cal_overestimates), np.max(PCC_cal_overestimates)), (np.min(PCC_nontrain_overestimates), np.max(PCC_nontrain_overestimates)))
    #plt.plot((-0.3, 0.3), (-0.3, 0.3))
    #plt.plot((np.min(PCC_cal_rmses), np.max(PCC_cal_rmses)), (np.min(PCC_nontrain_rmses), np.max(PCC_nontrain_rmses)))
    #plt.xlabel('PCC_cal_rmse')
    #plt.ylabel('PCC_nontrain_rmse')
    #plt.savefig('test.pdf')
    #plt.xlabel('PCC_cal_overestimate')
    #plt.ylabel('PCC_nontrain_overestimate')
    #plt.show()

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
