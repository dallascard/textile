import os
from optparse import OptionParser

import numpy as np
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
    patterns = fh.read_json(os.path.join(patterns_dir, 'bigrams.json'))

    counts_dict = {}

    for p, d in patterns.items():
        total = np.sum([v for k, v in d.items()])
        counts_dict[p] = total

    counts = Counter()
    counts.update(counts_dict)

    for p, count in counts.most_common(n=80):
        d = patterns[p]
        c = Counter()
        c.update(d)
        most_common = list(c.most_common(n=10))
        terms, counts = zip(*most_common)
        print('%s (%d): %s' % (p, count, ' '.join(terms[:10])))


if __name__ == '__main__':
    main()
