import os
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib as mpl
mpl.use('Agg')
import seaborn
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
    parser.add_option('--partition', dest='partition', default=None,
                      help='Partition for mfc (e.g. pre: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='model type [LR|MLP]: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type [l1|l2]: default=%default')
    parser.add_option('--dh', dest='dh', default=100,
                      help='Hidden dimension for MLP: default=%default')


    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    cshift = options.cshift
    n_train = str(int(options.n_train))
    n_calib = str(int(options.n_calib))
    sample_labels = options.sample
    base = options.base
    subset = options.subset
    partition = options.partition
    model_type = options.model
    penalty = options.penalty
    dh = str(int(options.dh))

    f1_f1s = []
    cal_f1s = []
    f1_maes = []
    cal_maes = []

    # basic LR f1: combining subset, label, repetitions, and pre/post date

    for objective in ['f1', 'calibration']:
        for label in ['label', 'Economic', 'Health', 'Legality', 'Political']:
            basename = '*_' + label + '_*_' + model_type + '_' + penalty
            if model_type == 'MLP':
                basename += '_' + dh
            basename += '_' + n_train + '_' + n_calib + '_' + objective
            if model_type == 'MLP' and base == 'mfc':
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

            #print(basename)
            files = glob(os.path.join('projects', base, subset, 'models', basename, 'results.csv'))
            files.sort()
            n_files = len(files)

            #print(files[0])
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

            label_maes = []
            for f in files[1:]:
                #print(f)
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

                label_maes.append(results.loc['PCC_nontrain', 'RMSE'])

            # repeat for accuracy / f1
            files = glob(os.path.join('projects', base, subset, 'models', basename, 'accuracy.csv'))
            files.sort()
            n_files = len(files)

            #print(files[0])
            results = fh.read_csv_to_df(files[0])
            df = results.copy()

            label_f1s = []
            for f in files[1:]:
                #print(f)
                results = fh.read_csv_to_df(f)
                df += results

                label_f1s.append(results.loc['cross_val', 'f1'])

            df = df / float(n_files)

            print(objective, label, np.mean(label_maes), np.mean(label_f1s))

            if objective == 'f1':
                f1_f1s.append(label_f1s)
                f1_maes.append(label_maes)
            else:
                cal_f1s.append(label_f1s)
                cal_maes.append(label_maes)


    df = pd.DataFrame(columns=['label', 'f1', 'objective', 'MAE'])

    label_list = ['Tone', 'Economics', 'Health', 'Legality', 'Politics']
    #df['objective'] = ['acc'] * 5 + ['cal'] * 5
    f1s = []
    maes = []
    labels = []
    objectives = []
    for group_i, group in enumerate(f1_f1s):
        n_samples = len(group)
        f1s.extend(group)
        labels.extend([label_list[group_i]] * n_samples)
        objectives.extend(['acc'] * n_samples)
    for group_i, group in enumerate(f1_maes):
        maes.extend(group)
    for group_i, group in enumerate(cal_f1s):
        n_samples = len(group)
        f1s.extend(group)
        labels.extend([label_list[group_i]] * n_samples)
        objectives.extend(['cal'] * n_samples)
    for group_i, group in enumerate(cal_maes):
        maes.extend(group)
    df['Label'] = labels
    df['f1'] = f1s
    df['objective'] = objectives
    df['MAE'] = maes


    fig, ax = plt.subplots()
    seaborn.boxplot(x='label', y='f1', hue='objective', data=df)
    fig.savefig('test.pdf')

    fig, ax = plt.subplots()
    seaborn.boxplot(x='label', y='MAE', hue='objective', data=df)
    fig.savefig('test2.pdf')

    """
    fig, ax = plt.subplots()
    for group_i, group in enumerate(f1_f1s):
        ax.scatter(np.ones_like(group) * group_i, group, c='blue')
    for group_i, group in enumerate(cal_f1s):
        ax.scatter(np.ones_like(group) * group_i + 0.2,  group, c='orange')
    fig.savefig('test.pdf', bbox_inches='tight')


    fig, ax = plt.subplots()
    for group_i, group in enumerate(f1_maes):
        ax.scatter(np.ones_like(group) * group_i, group, c='blue')
    for group_i, group in enumerate(cal_maes):
        ax.scatter(np.ones_like(group) * group_i + 0.2,  group, c='orange')
    fig.savefig('test2.pdf', bbox_inches='tight')
    """


if __name__ == '__main__':
    main()
