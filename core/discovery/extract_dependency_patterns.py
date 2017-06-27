import os
import re
import sys
from collections import Counter
from collections import defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn import preprocessing
from spacy.en import English

from ..util import file_handling as fh
from ..preprocessing import features
from ..util import dirs


def main():
    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--lemmatize', action="store_true", dest="lemmatize", default=False,
                      help='Use lemmas instead of words: default=%default')
    parser.add_option('--tags', action="store_true", dest="tags", default=False,
                      help='Use rich part of speech tags: default=%default')
    parser.add_option('--lower', action="store_true", dest="lower", default=False,
                      help='Lower case words: default=%default')
    #parser.add_option('--reference_subset', dest='reference', default=None,
    #                  help='Reference subset (i.e. train) for ensuring full : default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')
    lemmatize = options.lemmatize
    tags = options.tags
    lower = options.lower

    print("Reading data")
    data = fh.read_json(datafile)

    print("Loading spacy")
    parser = English()
    #tagger = phrasemachine.get_stdeng_nltk_tagger()

    labels = []
    unigrams = defaultdict(Counter)
    bigrams = defaultdict(Counter)
    #trigrams = defaultdict(Counter)

    print("Parsing texts")
    keys = list(data.keys())
    keys.sort()

    for k_i, key in enumerate(keys):
        if k_i % 100 == 0 and k_i > 0:
            print(k_i)

        item = data[key]
        text = item['text']
        # replace underscores with dashes to avoid confusion
        text = re.sub('_', '-', text)

        # parse the text with spaCy
        parse = parser(text)

        for token in parse:
            word = get_word(token, lemmatize=lemmatize, lower=lower)
            pos = get_pos_tag(token, tags=tags)
            dep = token.dep_
            parent = token.head
            parent_word = get_word(parent, lemmatize=lemmatize, lower=lower)
            parent_pos = get_pos_tag(parent, tags=tags)
            dep_pattern = '_'.join([parent_pos, dep, pos])

            #children_tokens = token.children
            #ent_type = token.ent_type_
            #prefix = token.prefix_
            #suffix = token.suffix_

            unigrams[pos].update([word])
            bigrams[dep_pattern].update([' '.join([parent_word, word])])


    output_dir = dirs.dir_patterns(project_dir, subset)
    fh.makedirs(output_dir)
    fh.write_to_json(unigrams, os.path.join(output_dir, 'unigrams.json'))


def get_word(token, lemmatize=False, lower=False):
    #  get word and remove whitespace
    if lemmatize:
        word = re.sub('\s', '', token.lemma_)
    else:
        word = re.sub('\s', '', token.orth_)
    if lower:
        word = word.lower()
    return word


def get_pos_tag(token, tags=False):
    if tags:
        return token.tag_
    else:
        return token.pos_


if __name__ == '__main__':
    main()
