import os
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog train.csv test.csv project_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='prop', default=1.0,
                      help='Use only a random proportion of training data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    train_file = args[0]
    test_file = args[1]
    project_dir = args[2]
    prop = float(options.prop)

    import_sentiment140(train_file, test_file, project_dir, prop)


def import_sentiment140(train_file, test_file, project_dir, prop=1.0):
    print("Loading data")
    train = load_df(train_file)
    test = load_df(test_file)

    n_train, _ = train.shape
    n_test, _ = test.shape

    print(n_train)
    #print(len(set(train.id)))

    # make compatible indices
    test.index = list(range(n_train, n_train + n_test))

    train['test'] = 0
    test['test'] = 1

    print("Converting to JSON")
    train_dict = convert_to_json(train, prop)
    test_dict = convert_to_json(test)

    print("Saving data")
    data_dir = dirs.dir_data_raw(project_dir)
    fh.makedirs(data_dir)

    fh.write_to_json(train_dict, os.path.join(data_dir, 'train.json'))
    fh.write_to_json(test_dict, os.path.join(data_dir, 'test.json'))


def load_df(filename):
    df = fh.read_csv_to_df(filename, index_col=None, header=None, encoding='Windows-1252')
    print(df.head())
    cols = ['label', 'id', 'date_string', 'query', 'user', 'text']
    df.columns = cols
    df['date'] = [pd.Timestamp(d) for d in df.date_string]
    return df


def convert_to_json(df, prop=1.0):
    dayofyears = Counter()

    data = {}
    index = list(df.index)
    n_items = len(index)
    if prop < 1.0:
        subset_size = int(prop * n_items)
        subset = np.random.choice(range(n_items), size=subset_size, replace=False)
        index = [index[i] for i in subset]
        print("Using a random subset of %d tweets" % subset_size)

    for i in index:
        row = df.loc[i]
        data[str(i)] = {'labels': {'positive': int(row.label)},
                        'id': int(row.id),
                        'date': str(row.date),
                        'year': row.date.year,
                        'month': row.date.month,
                        'dayofyear': row.date.dayofyear,
                        'text': row.text,
                        'user': row.user,
                        'query': row.query,
                        'hour': row.date.hour,
                        'hour4': row.date.hour // 6}
        dayofyears.update([row.date.dayofyear])

    print(dayofyears.most_common())
    return data


if __name__ == '__main__':
    main()
