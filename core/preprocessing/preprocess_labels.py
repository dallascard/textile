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

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    if len(args) > 2:
        metadata_fields = args[2].split(',')
    else:
        metadata_fields = []
    label_name = options.label

    preprocess_labels(project_dir, subset, label_name, metadata_fields)


def preprocess_labels(project_dir, subset, label_name, metadata_fields):

    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())
    keys.sort()
    items = []

    # first determine the number of classes
    label_set = set()
    for k_i, key in enumerate(keys):
        if k_i % 10000 == 0 and k_i > 0:
            print(k_i)
        item = data[key]
        if label_name in item:
            labels = item[label_name]
        else:
            labels = 0
        if type(labels) == dict:
            label_set.update(list(labels.keys()))
        else:
            label_set.add(labels)

        if 'name' in item:
            name = item['name']
        else:
            name = str(key)
        items.append(name)

    n_classes = len(label_set)
    print("Found %d classes" % n_classes)

    # make label index
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

    # now extract the label and metadata for each item
    metadata_lists = {}
    for m in metadata_fields:
        metadata_lists[m] = []

    #labels_df = pd.DataFrame(np.zeros((len(keys), n_classes), dtype=int), index=items, columns=np.arange(n_classes))
    labels_matrix = np.zeros((len(keys), n_classes), dtype=int)

    for k_i, key in enumerate(keys):
        if k_i % 10000 == 0 and k_i > 0:
            print(k_i)

        item = data[key]

        # TODO: make this faster; avoid inserting into dataframe
        #labels = item[label_name]
        if label_name in item:
            labels = item[label_name]
        else:
            labels = 0

        if type(labels) == dict:
            for k, v in labels.items():
                labels_matrix[k_i, label_index[k]] = v
        else:
            labels_matrix[k_i, label_index[labels]] = 1

        for m in metadata_fields:
            metadata_lists[m].append(item[m])

    labels_df = pd.DataFrame(labels_matrix, index=items, columns=np.arange(n_classes))

    # nah, do this in training...
    # normalize those rows that are unanimous
    #print("Normalizing")
    #not_unan = np.array(np.sum(labels_df.values > 0, axis=1) != 1)
    #row_sum = np.reshape(np.sum(labels_df.values, axis=1), (len(not_unan), 1))
    #row_sum[not_unan] = 1.0
    #labels_df = pd.DataFrame(np.array(labels_df.values / row_sum, dtype=int), index=labels_df.index, columns=labels_df.columns)

    print("Saving labels")
    output_dir = dirs.dir_labels(project_dir, subset)
    fh.makedirs(output_dir)
    labels_df.to_csv(os.path.join(output_dir, label_name + '.csv'))
    fh.write_to_json(label_index, os.path.join(output_dir, label_name + '_index.json'))

    if len(metadata_fields) > 0:
        print("Saving metadata")
        metadata = pd.DataFrame([], index=labels_df.index)
        for m in metadata_fields:
            metadata[m] = metadata_lists[m]
        output_dir = dirs.dir_subset(project_dir, subset)
        metadata.to_csv(os.path.join(output_dir, 'metadata.csv'))


if __name__ == '__main__':
    main()
