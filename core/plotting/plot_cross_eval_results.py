import os
from optparse import OptionParser

from ..util import file_handling as fh
from ..util import dirs

def main():
    usage = "%prog project_dir subset cross_field_name model_basename"

    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()


    project_dir = args[0]
    subset = args[1]
    field_name = args[2]
    model_basename = args[3]

    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata[field_name].values))

    field_vals.sort()

    for v_i, v in enumerate(field_vals):




if __name__ == '__main__':
    main()
