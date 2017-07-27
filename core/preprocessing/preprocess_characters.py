import os
import re
import sys
import gensim
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn import preprocessing
from spacy.en import English

from ..util import file_handling as fh
from ..preprocessing import normalize_text, features
from ..util import dirs


def main():
    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--lower', action="store_true", dest="lower", default=False,
                      help='Lower case the text: default=%default')
    parser.add_option('-n', dest='ngrams', default=3,
                      help='Max degree of character n-grams: default=%default')
    parser.add_option('-d', dest='display', default=1000,
                      help='Display progress every X items: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    lower = options.lower
    ngrams = int(options.ngrams)
    display = int(options.display)

    preprocess_chracters(project_dir, subset, lower, ngrams, display)


def preprocess_chracters(project_dir, subset, lower, ngrams, display):

    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())
    keys.sort()

    items = []

    chargrams = {}
    chargram_vocab = set()

    for k_i, key in enumerate(keys):
        if k_i % display == 0 and k_i > 0:
            print(k_i)

        item = data[key]
        if 'name' in item:
            name = item['name']
        else:
            name = str(key)
        items.append(name)

        text = item['text']

        if lower:
            text = text.lower()

        # replace underscores with dashes to avoid confusion
        text = re.sub('_', '-', text)

        if ngrams > 0:
            letters = list(text)
            counter = Counter()
            for c in range(1, ngrams+1):
                counter.update([''.join(letters[i:i+c]) for i in range(len(letters)-c+1)])
            chargrams[name] = dict(counter)

    if ngrams > 0:
        chargram_feature = features.create_from_dict_of_counts('chargrams', chargrams)
        chargram_feature.save_feature(dirs.dir_features(project_dir, subset))


if __name__ == '__main__':
    main()
