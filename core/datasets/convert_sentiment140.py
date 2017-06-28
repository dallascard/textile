import os
from optparse import OptionParser

import pandas as pd

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog train.csv test.csv output_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    train_file = args[0]
    test_file = args[1]
    output_dir = args[2]

    convert_sentiment140(train_file, test_file, output_dir)


def convert_sentiment140(train_file, test_file, output_dir):
    print("Loading data")
    train = load_df(train_file)
    test = load_df(test_file)

    n_train, _ = train.shape
    n_test, _ = test.shape

    print(n_train)
    print(len(set(train.id)))

    # make compatible indices
    test.index = list(range(n_train, n_train + n_test))

    train['test'] = 0
    test['test'] = 1

    print("Converting to JSON")
    train_dict = convert_to_json(train)
    test_dict = convert_to_json(test)

    print("Saving data")
    data_dir = dirs.dir_data_raw(output_dir)
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


def convert_to_json(df):
    data = {}
    for i in df.index:
        row = df.loc[i]
        data[str(i)] = {'label': int(row.label),
                        'id': int(row.id),
                        'date': str(row.date),
                        'year': row.date.year,
                        'month': row.date.month,
                        'dayofyear': row.date.dayofyear,
                        'text': row.text,
                        'user': row.user,
                        'query': row.query}
    return data


if __name__ == '__main__':
    main()
