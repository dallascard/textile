import os
from optparse import OptionParser

import pandas as pd

from ..util import dirs
from ..util import file_handling as fh

def main():
    usage = "%prog reviews_file.json project_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    reviews_file = args[0]
    project = args[1]

    import_review_data(reviews_file, project)


def import_review_data(reviews_file, project_dir):
    print("Loading data")
    reviews = fh.read_json_lines(reviews_file)

    n_items = len(reviews)

    print("Loaded %d items" % n_items)

    dates = pd.DataFrame(columns=['date'])

    data = {}
    for k_i, k in enumerate(reviews.keys()):
        review = reviews[k]
        if k_i % 1000 == 0:
            print(k_i)
        helpfulness = review['helpful']
        n_helpful_votes = helpfulness[0]
        n_votes = helpfulness[1]
        if n_votes > 0:
            data[k] = {}
            data[k]['reviewerID'] = review['reviewerID']
            data[k]['text'] = review['reviewText']
            data[k]['rating'] = review['overall']
            data[k]['summary'] = review['summary']
            data[k]['label'] = {'helpful': n_helpful_votes, 'not': n_votes - n_helpful_votes}
            date_string = review['reviewTime']
            parts = date_string.split(',')
            year = int(parts[1])
            parts2 = parts[0].split()
            month = int(parts2[0])
            day = int(parts2[1])
            data[k]['year'] = year
            data[k]['month'] = month
            data[k]['day'] = day
            date = pd.Timestamp(year=year, month=month, day=day)
            dates.loc[k] = date

    print(dates.date.min())
    print(dates.date.max())

    print("Saving data")
    data_dir = dirs.dir_data_raw(project_dir)
    fh.makedirs(data_dir)

    fh.write_to_json(data, os.path.join(data_dir, 'all.json'))

if __name__ == '__main__':
    main()
