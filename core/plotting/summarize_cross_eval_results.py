import os
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import file_handling as fh
from ..util import dirs

def main():
    usage = "%prog project_dir subset cross_field_name"

    parser = OptionParser(usage=usage)
    parser.add_option('--prefix', dest='prefix', default=None,
                      help='Prefix for name before subset_fieldname: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    field_name = args[2]
    model_basename = subset + '_' + field_name

    prefix = options.prefix
    if prefix is not None:
        model_basename = prefix + model_basename

    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata[field_name].values))

    field_vals.sort()
    n_field_vals = len(field_vals)

    #methods = ['train', 'calib', 'CC', 'PCC', 'ACC', 'ACC_int', 'PVC', 'PVC_int', 'Venn']
    methods = ['nontest', 'CC', 'PCC', 'ACC_int', 'PVC_int', 'Venn']
    columns = ['N'] + methods
    mean_rmse_df = pd.DataFrame([], columns=columns)
    min_rmse_df = pd.DataFrame([], columns=columns)
    max_rmse_df = pd.DataFrame([], columns=columns)
    best_counts_df = pd.DataFrame(np.zeros(len(methods)), index=methods, columns=['best'])
    worst_counts_df = pd.DataFrame(np.zeros(len(methods)), index=methods, columns=['worst'])

    test_estimate_pairs = []

    print(field_vals)
    for v_i, v in enumerate(field_vals):

        model_name = model_basename + '_' + str(v) + '_*'
        output_files = glob(os.path.join(dirs.dir_models(project_dir), model_name, field_name + '_' + str(v) + '.csv'))
        output_files.sort()

        errors_df = pd.DataFrame([], columns=columns)

        for f_i, f in enumerate(output_files):
            df = pd.read_csv(f, index_col=0, header=0)
            print(df)
            N = df.loc['nontest', 'N']
            errors = df['RMSE'].values
            errors_df.loc[f_i] = np.r_[N, errors[1:]]
            test_estimate_pairs.append((df.loc['nontest', 'N'], df.loc['test', 'estimate']))
            best_counts_df['best'] += errors[1:] <= np.min(errors[1:])
            worst_counts_df['worst'] += errors[1:] >= np.max(errors[1:])

        mean_rmse_df.loc[v] = errors_df.mean(axis=0)
        min_rmse_df.loc[v] = errors_df.min(axis=0)
        max_rmse_df.loc[v] = errors_df.max(axis=0)

    for m_i, m in enumerate(methods):
        print(m)
        print(mean_rmse_df[m])
        #for i, loc in enumerate(mean_rmse_df.index):
        #    ax.plot([i + m_i * offset, i + m_i * offset], [min_rmse_df.loc[loc, m], max_rmse_df.loc[loc, m]], c='k', label=None, alpha=0.7)
        #ax.scatter(np.arange(n_field_vals) + m_i * offset, mean_rmse_df[m].values, label=m)

    print(best_counts_df)
    print(worst_counts_df)

if __name__ == '__main__':
    main()
