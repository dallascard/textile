from optparse import OptionParser

from ..util import dirs
from ..util import file_handling as fh

def main():
    usage = "%prog project_name data.json metadata.json"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    data_file = args[1]
    metadata_file = args[2]

    convert_mfc(project, data_file, metadata_file)

def convert_mfc(project, data_file, metadata_file):
    fh.makedirs(dirs.dir_data_raw(project))

    data = fh.read_json(data_file)



if __name__ == '__main__':
    main()
