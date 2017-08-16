import os
import re
from collections import defaultdict
from optparse import OptionParser

import numpy as np

from ..util import file_handling as fh
from ..util import dirs

def main():
    usage = "%prog input_dir project_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('--subset', dest='subset', default='article',
                      help='article or user: default=%default')
    #parser.add_option('--approx', action="store_true", dest="approx", default=False,
    #                  help='Approximate label distribution: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    project = args[1]

    subset = options.subset

    import_data(input_dir, project, subset)


def import_data(input_dir, project, subset):
    print("Loading data")
    comments_file = os.path.join(input_dir, 'attack_annotated_comments.tsv')
    comments = fh.read_csv_to_df(comments_file, sep='\t')

    annotations_file = os.path.join(input_dir, 'attack_annotations.tsv')
    annotations = fh.read_csv_to_df(annotations_file, sep='\t')

    n_items, _ = comments.shape
    n_annotations, _ = annotations.shape
    print("Loaded %d items" % n_items)
    print("Loaded %d annotations" % n_annotations)

    year_set = set()
    ns_set = set()
    sample_set = set()
    split_set = set()

    data = {}
    print("Processing comments")
    for i, item in enumerate(comments.index):
        year_set.add(comments.loc[item, 'year'])
        ns_set.add(comments.loc[item, 'ns'])
        sample_set.add(comments.loc[item, 'sample'])
        split_set.add(comments.loc[item, 'split'])
        if comments.loc[item, 'sample'] == 'random' and comments.loc[item, 'ns'] == subset:
            data[item] = {}
            data[item]['text'] = re.sub('NEWLINE_TOKEN', '\n', comments.loc[item, 'comment'])
            data[item]['year'] = int(comments.loc[item, 'year'])
            data[item]['split'] = comments.loc[item, 'split']
            data[item]['label'] = {0: 0,  1: 0}

    print("Processing annotations")
    items = list(annotations.index)
    ratings = annotations['attack'].values

    for i, item in enumerate(items):
        if item in data:
            data[item]['label'][ratings[i]] += 1

    print(year_set)
    print(ns_set)
    print(sample_set)
    print(split_set)

    year_averages = defaultdict(list)
    for item, values in data.items():
        year = values['year']
        attack = values['label'][0] / float(values['label'][0] + values['label'][1])
        year_averages[year].append(attack)

    years = list(year_averages.keys())
    years.sort()
    attacks = [np.mean(year_averages[year]) for year in years]
    n_items = [len(year_averages[year]) for year in years]
    for i, year in enumerate(years):
        print("%d, %d, %0.4f" % (year, n_items[i], attacks[i]))

    print("Saving %d items" % len(data))
    data_dir = dirs.dir_data_raw(project)
    fh.makedirs(data_dir)
    fh.write_to_json(data, os.path.join(data_dir, subset + '.json'))


if __name__ == '__main__':
    main()
