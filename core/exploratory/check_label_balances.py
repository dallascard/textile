import os
import glob
from optparse import OptionParser

from core.util import file_handling as fh
from core.util import dirs


def main():
    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--field', dest='field', default='year',
                      help='Field on which to split for train and test: default=%default')
    parser.add_option('--test_start', dest='test_start', default=2011,
                      help='Use training data from before this field value: default=%default')
    parser.add_option('--test_end', dest='test_end', default=2012,
                      help='Last field value of test data to use: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]

    field = options.field
    test_start = int(options.test_start)
    test_end = int(options.test_end)

    label_files = glob.glob(os.path.join(dirs.dir_labels(project_dir, subset), '*.csv'))

    for label_file in label_files:
        label = os.path.basename(label_file).split('.')[0]
        check_balances(project_dir, subset, field, test_start, test_end, label)


def check_balances(project_dir, subset, field, test_start, test_end, label):

    # load the file that contains metadata about each item
    metadata_file = os.path.join(dirs.dir_subset(project_dir, subset), 'metadata.csv')
    metadata = fh.read_csv_to_df(metadata_file)
    field_vals = list(set(metadata[field].values))
    field_vals.sort()
    print("Splitting data according to %s", field)
    print("Values:", field_vals)

    print("\nTesting on %s to %s" % (test_start, test_end))

    # first, split into training and non-train data based on the field of interest
    all_items = list(metadata.index)
    test_selector_all = (metadata[field] >= int(test_start)) & (metadata[field] <= int(test_end))
    test_subset_all = metadata[test_selector_all]
    test_items_all = test_subset_all.index.tolist()
    n_test_all = len(test_items_all)

    train_selector_all = metadata[field] < int(test_start)
    train_subset_all = metadata[train_selector_all]
    train_items_all = list(train_subset_all.index)
    n_train_all = len(train_items_all)

    print("Train: %d, Test: %d (labeled and unlabeled)" % (n_train_all, n_test_all))

    # load all labels
    label_dir = dirs.dir_labels(project_dir, subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    # add in a stage to eliminate items with no labels
    print("Subsetting items with labels")
    label_sums_df = labels_df.sum(axis=1)
    labeled_item_selector = label_sums_df > 0
    labels_df = labels_df[labeled_item_selector]
    n_labeled_items, n_classes = labels_df.shape
    labeled_items = set(labels_df.index)

    train_items_labeled = [i for i in train_items_all if i in labeled_items]
    test_items = [i for i in test_items_all if i in labeled_items]

    print(label, labels_df.loc[train_items_labeled].mean(axis=0), labels_df.loc[test_items].mean(axis=0))


if __name__ == '__main__':
    main()