import os
import re
from optparse import OptionParser

from ..util import file_handling as fh
from ..util import dirs

def main():
    usage = "%prog input_dir project_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('-p', dest='prop', default=1.0,
    #                  help='Use only a random proportion of training data: default=%default')
    #parser.add_option('--approx', action="store_true", dest="approx", default=False,
    #                  help='Approximate label distribution: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    project = args[1]
    #prop = float(options.prop)
    prop = 1.0

    import_data(input_dir, project, prop)


def import_data(input_dir, project, prop):
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
    for i, item in comments.index:
        year_set.add(comments.loc[item, 'year'])
        ns_set.add(comments.loc[item, 'ns'])
        sample_set.add(comments.loc[item, 'sample'])
        split_set.add(comments.loc[item, 'split'])
        if comments.loc[item, 'sample'] == 'random':
            data[item] = {}
            data[item]['text'] = re.sub('NEWLINE_TOKEN', '\n', comments.loc[item, 'comment'])
            data[item]['year'] = int(comments.loc[item, 'year'])
            data[item]['split'] = comments.loc[item, 'split']
            data[item]['label'] = {0: 0,  1: 0}

    for i, item in annotations.index:
        attack = int(annotations.loc[item, 'attack'])
        if item in data:
            data[item]['label'][attack] += 1

    print(year_set)
    print(ns_set)
    print(sample_set)
    print(split_set)

    data_dir = dirs.dir_data_raw(project)
    fh.makedirs(data_dir)
    fh.write_to_json(data, os.path.join(data_dir, 'all.json'))


if __name__ == '__main__':
    main()
