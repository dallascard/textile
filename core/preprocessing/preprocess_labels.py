import os
from optparse import OptionParser

import numpy as np
import pandas as pd

from ..util import file_handling as fh
from ..util import dirs


def main():
    """Preprocess labels and metadata (convert them to my format)"""

    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Name of label (in data_file.json): default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    label_name = options.label
    preprocess_labels(project_dir, subset, label_name)


def preprocess_labels(project_dir, subset, label_name):

    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())
    keys.sort()

    # make a list of metadata fields (not text or label)
    fields = list(data[keys[0]].keys())
    print("Fields found in data:", fields)
    fields.remove('text')
    fields.remove(label_name)

    metadata = pd.DataFrame(columns=fields)
    labels = []
    label_set = set()

    for k_i, key in enumerate(keys):
        if k_i % 1000 == 0 and k_i > 0:
            print(k_i)

        item = data[key]
        #if type(item[label_name]) == dict:
        #    label_dict = item[label_name]
        #else:
        #    label_dict = {item[label_name]: 1}
        #label_set.update(label_dict.keys())

        label = item[label_name]
        label_set.add(label)
        labels.append(label)

        if 'name' in item:
            name = item['name']
        else:
            name = str(key)
        metadata.loc[name] = [item[f] for f in fields]

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

    labels_df = pd.DataFrame(int_labels, index=keys, columns=[label_name])

    output_dir = dirs.dir_labels(project_dir, subset)
    fh.makedirs(output_dir)
    labels_df.to_csv(os.path.join(output_dir, label_name + '.csv'))
    fh.write_to_json(label_index, os.path.join(output_dir, label_name + '_index.json'))
    fh.write_to_json(int_labels, os.path.join(output_dir, label_name + '.json'))

    print("Saving metadata")
    output_dir = dirs.dir_subset(project_dir, subset)
    metadata.to_csv(os.path.join(output_dir, 'metadata.csv'))


if __name__ == '__main__':
    main()
