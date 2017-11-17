import os
from optparse import OptionParser

from collections import Counter

import numpy as np
import pandas as pd

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog reviews_file.json project_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='prop', default=1.0,
                      help='Use only a random proportion of training data: default=%default')

    (options, args) = parser.parse_args()

    reviews_file = args[0]
    project = args[1]
    prop = float(options.prop)

    import_review_data(reviews_file, project, prop)


def import_review_data(reviews_file, project_dir, prop):
    print("Loading data")
    reviews = fh.read_json_lines(reviews_file)

    n_items = len(reviews)

    print("Loaded %d items" % n_items)

    dates = pd.DataFrame(columns=['date'])
    keys = list(reviews.keys())
    reviewers = set()
    asins = set()
    year_counts = Counter()

    if prop < 1.0:
        subset_size = int(prop * n_items)
        subset = np.random.choice(range(n_items), size=subset_size, replace=False)
        keys = [keys[i] for i in subset]
        print("Using a random subset of %d reviews" % subset_size)

    ratings = Counter()

    data = {}
    for k_i, k in enumerate(keys):
        review = reviews[k]
        if k_i % 1000 == 0:
            print(k_i)
        helpfulness = review['helpful']
        n_helpful_votes = helpfulness[0]
        n_votes = helpfulness[1]
        if n_votes > 0:
            date_string = review['reviewTime']
            parts = date_string.split(',')
            year = int(parts[1])
            parts2 = parts[0].split()
            month = int(parts2[0])
            day = int(parts2[1])
            if year > 2006:
                data[k] = {}
                data[k]['reviewerID'] = review['reviewerID']
                data[k]['asin'] = review['asin']
                asins.add(review['asin'])
                reviewers.add(review['reviewerID'])
                data[k]['text'] = review['summary'] + '\n\n' + review['reviewText']
                data[k]['rating'] = review['overall']
                ratings.update([int(review['overall'])])
                data[k]['summary'] = review['summary']
                data[k]['labels'] = {'helpfulness': {0: n_votes - n_helpful_votes,  1: n_helpful_votes}}
                year_counts.update([year])
                data[k]['year'] = year
                data[k]['month'] = month
                data[k]['day'] = day
                date = pd.Timestamp(year=year, month=month, day=day)
                dates.loc[k] = date

    print("Found %d reviews with at least one vote" % len(data))

    print(ratings.items())

    print("Earliest date:", dates.date.min())
    print("Latest date:", dates.date.max())
    print("%d reviewers" % len(reviewers))
    print("%d products" % len(asins))
    print(year_counts)

    print("Saving data")
    data_dir = dirs.dir_data_raw(project_dir)
    fh.makedirs(data_dir)
    fh.write_to_json(data, os.path.join(data_dir, 'all.json'))


if __name__ == '__main__':
    main()
