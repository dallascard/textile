import os
from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import file_handling as fh
from ..util import dirs


def main():
    """Preprocess labels and metadata (convert them to my format)"""

    usage = "%prog project_dir subset [metadata1,metadata2,...]"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Name of label (in data_file.json): default=%default')
    parser.add_option('-d', dest='display', default=1000,
                      help='Display progress every X items: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    if len(args) > 2:
        metadata_fields = args[2].split(',')
    else:
        metadata_fields = []
    label_name = options.label
    display = int(options.display)
    preprocess_labels(project_dir, subset, label_name, metadata_fields, display)


def preprocess_labels(project_dir, subset, label_name, metadata_fields, display):

    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())
    keys.sort()

    labels = []
    label_set = set()

    metadata_lists = {}
    for m in metadata_fields:
        metadata_lists[m] = []

    items = []

    for k_i, key in enumerate(keys):
        if k_i % display == 0 and k_i > 0:
            print(k_i)

        item = data[key]

        if 'name' in item:
            name = item['name']
        else:
            name = str(key)
        items.append(name)

        # TODO: eventually change this to keeping a sparse dictionary of label counts
        label = item[label_name]
        label_set.add(label)
        labels.append(label)

        for m in metadata_fields:
            metadata_lists[m].append(item[m])

    print("Saving labels")
    label_set = list(label_set)
    try:
        if np.all([label.isdigit for label in label_set]):
            label_index = {label: int(label) for label in label_set}
        else:
            label_set.sort()
            label_index = {label: i for i, label in enumerate(label_set)}
    except ValueError:    # for sting labels
        label_set.sort()
        label_index = {label: i for i, label in enumerate(label_set)}
    except AttributeError:  # for float labels
        label_set.sort()
        label_index = {label: i for i, label in enumerate(label_set)}

    int_labels = [label_index[label] for label in labels]
    #int_labels = {k: {label_index[label]: value for label, value in item_labels.items()} for k, item_labels in labels.items()}

    labels_df = pd.DataFrame(int_labels, index=items, columns=[label_name])

    output_dir = dirs.dir_labels(project_dir, subset)
    fh.makedirs(output_dir)
    labels_df.to_csv(os.path.join(output_dir, label_name + '.csv'))
    fh.write_to_json(label_index, os.path.join(output_dir, label_name + '_index.json'))
    fh.write_to_json(int_labels, os.path.join(output_dir, label_name + '.json'))

    if len(metadata_fields) > 0:
        print("Saving metadata")
        metadata = pd.DataFrame([], index=items)
        for m in metadata_fields:
            metadata[m] = metadata_lists[m]
        output_dir = dirs.dir_subset(project_dir, subset)
        metadata.to_csv(os.path.join(output_dir, 'metadata.csv'))


if __name__ == '__main__':
    main()
