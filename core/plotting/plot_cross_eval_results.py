import os
from glob import glob
from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..util import file_handling as fh
from ..util import dirs

def main():
    usage = "%prog project_dir subset cross_field_name"

    parser = OptionParser(usage=usage)
    parser.add_option('--offset', dest='offset', default=0.1,
                      help='Keyword argument: default=%default')
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
        model_basename = prefix + '_' + model_basename

    offset = float(options.offset)

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
    test_estimate_pairs = []

    for v_i, v in enumerate(field_vals):

        model_name = model_basename + '_' + str(v) + '_*'
        output_files = glob(os.path.join(dirs.dir_models(project_dir), model_name, field_name + '_' + str(v) + '.csv'))
        output_files.sort()

        errors_df = pd.DataFrame([], columns=columns)

        for f_i, f in enumerate(output_files):
            df = pd.read_csv(f, index_col=0, header=0)
            #N = df.loc['calibration', 'N']
            N = df.loc['nontest', 'N']
            errors = df['RMSE'].values
            errors_df.loc[f_i] = np.r_[N, errors[1:]]
            #test_estimate_pairs.append((df.loc['calibration', 'N'], df.loc['test', 'estimate']))
            test_estimate_pairs.append((df.loc['nontest', 'N'], df.loc['test', 'estimate']))

        mean_rmse_df.loc[v] = errors_df.mean(axis=0)
        min_rmse_df.loc[v] = errors_df.min(axis=0)
        max_rmse_df.loc[v] = errors_df.max(axis=0)

    fig, ax = plt.subplots()
    for m_i, m in enumerate(methods):
        for i, loc in enumerate(mean_rmse_df.index):
            ax.plot([i + m_i * offset, i + m_i * offset], [min_rmse_df.loc[loc, m], max_rmse_df.loc[loc, m]], c='k', label=None, alpha=0.7)
        ax.scatter(np.arange(n_field_vals) + m_i * offset, mean_rmse_df[m].values, label=m)

    #test_Ns, test_estimates = list(zip(*test_estimate_pairs))
    #ax.scatter(test_Ns, test_estimates, c='k', label='test')

    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
