import os
from optparse import OptionParser

import numpy as np
#import matplotlib.pyplot as plt

from ..util import dirs
from ..util import file_handling as fh

def main():
    usage = "%prog project subset field_name"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--n_classes', dest='n_classes', default=None,
                      help='Number of classes [None=auto]: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    subset = args[1]
    field_name = args[2]
    label = options.label
    n_classes = options.n_classes
    if n_classes is not None:
        n_classes = int(n_classes)

    datafile = os.path.join(dirs.dir_data_raw(project), subset + '.json')
    data = fh.read_json(datafile)

    field_vals = list(set([data[k][field_name] for k in data.keys()]))
    field_vals.sort()

    label_set = list(set([data[k][label] for k in data.keys()]))
    label_set.sort()
    if n_classes is None:
        n_classes = len(label_set)
    label_vals = range(n_classes)
    label_index = dict(zip(label_set, label_vals))

    proportions = np.zeros([len(field_vals), len(label_vals)])
    n_stories = []

    for v_i, val in enumerate(field_vals):
        print(val)
        subset = {k: v for k, v in data.items() if data[k][field_name] == val}
        n_stories.append(len(subset))
        label_counts = np.bincount([label_index[subset[k][label]] for k in subset.keys()], minlength=n_classes)
        label_props = label_counts / float(len(subset))
        proportions[v_i, :] = label_props

    print(proportions)
    print(n_stories)

if __name__ == '__main__':
    main()
