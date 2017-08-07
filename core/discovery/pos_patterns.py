import os
import re
import sys
import gensim
from collections import Counter, defaultdict
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
    parser.add_option('--lemmatize', action="store_true", dest="lemmatize", default=False,
                      help='Use lemmas instead of words: default=%default')
    parser.add_option('-n', dest='ngrams', default=2,
                      help='Max degree of word n-grams [1 or 2]: default=%default')
    parser.add_option('--fast', action="store_true", dest="fast", default=False,
                      help='Only do things that are fast (i.e. only splitting, no parsing, no lemmas): default=%default')
    parser.add_option('-d', dest='display', default=100,
                      help='Display progress every X items: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')
    lower = options.lower
    lemmatize = options.lemmatize
    fast = options.fast
    ngrams = int(options.ngrams)
    display = int(options.display)

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())
    keys.sort()

    if not fast:
        print("Loading spacy")
        parser = English()

    items = []
    unigrams = defaultdict(Counter)
    bigrams = defaultdict(Counter)

    print("Parsing text")

    for k_i, key in enumerate(keys[:2]):
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

        if not fast:
            # parse the text with spaCy
            parse = parser(text)

            for token in parse:
                word = token.orth_
                shape = token.shape_
                lemma = token.lemma_
                pos = token.pos_
                tag = token.tag_
                #dep = token.dep_
                #head_token = token.head
                #children_tokens = token.children
                #ent_type = token.ent_type_
                #prefix = token.prefix_
                #suffix = token.suffix_

                unigrams[pos].update([word])

            for sent in parse.sents:
                if len(sent) > 1:
                    word_pos_pairs = [get_word_pos_pair(sent[i]) for i in range(len(sent))]
                    words, tags = zip(*word_pos_pairs)
                    for i in range(len(sent)-1):
                        key = tags[i] + '_' + tags[i+1]
                        bigrams[key].update([words[i] + '_' + words[i+1]])

    keys = bigrams.keys()
    order = list(np.argsort([len(bigrams[k]) for k in keys]).tolist())
    order.reverse()

    k0 = keys[order[0]]
    first = defaultdict(int)
    second = defaultdict(int)
    for bigram in bigrams[k0]:
        parts = bigram.split('_')
        first.update([parts[0]])
        second.update([parts[1]])


    #print("Creating word features")
    #word_feature = features.create_from_dict_of_counts('unigrams', unigrams)
    #word_feature.save_feature(dirs.dir_features(project_dir, subset))

    #if ngrams > 1:
    #    bigram_feature = features.create_from_dict_of_counts('bigrams', bigrams)
    #    bigram_feature.save_feature(dirs.dir_features(project_dir, subset))



def get_word_pos_pair(token):
    # get lemma and remove whitespace
    word = re.sub('\s', '', token.orth_)
    return word, token.pos_



if __name__ == '__main__':
    main()
