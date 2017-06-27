import os
from optparse import OptionParser

from collections import Counter

from ..util import dirs
from ..util import file_handling as fh

def main():
    usage = "%prog project subset "
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()
    project_dir = args[0]
    subset = args[1]

    patterns_dir = dirs.dir_patterns(project_dir, subset)
    patterns = fh.read_json(os.path.join(patterns_dir, 'unigrams.json'))

    for p, d in patterns.items():
        c = Counter()
        c.update(d)
        most_common = c.most_common(n=10)
        print(p)
        print(most_common)


if __name__ == '__main__':
    main()
