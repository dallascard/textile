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
    parser.add_option('--approx', action="store_true", dest="approx", default=False,
                      help='Approximate label distribution: default=%default')


    (options, args) = parser.parse_args()

    reviews_file = args[0]
    project = args[1]
    prop = float(options.prop)
    approx = options.approx

    import_review_data(reviews_file, project, prop, approx)


def import_review_data(reviews_file, project_dir, prop, approx=False):
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
                data[k]['text'] = review['reviewText']
                data[k]['rating'] = review['overall']
                data[k]['summary'] = review['summary']
                if approx:
                    helpfulness_prop = n_helpful_votes / n_votes
                    n_helpful = int(np.round(helpfulness_prop * 10))
                    n_not_helpful = 10 - n_helpful
                    data[k]['label'] = {0: n_not_helpful,  1: n_helpful}
                else:
                    data[k]['label'] = {0: n_votes - n_helpful_votes,  1: n_helpful_votes}
                year_counts.update([year])
                data[k]['year'] = year
                data[k]['month'] = month
                data[k]['day'] = day
                date = pd.Timestamp(year=year, month=month, day=day)
                dates.loc[k] = date

    print("Found %d reviews with at least one vote" % len(data))

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
