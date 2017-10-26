import os
import re
from collections import Counter
from optparse import OptionParser

import numpy as np
from spacy.en import English

from ..util import file_handling as fh
from ..preprocessing import normalize_text, features
from ..util import dirs


def main():
    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--suffix', dest='suffix', default="",
                      help='Suffix for file name (after unigrams/bigrams): default=%default')
    parser.add_option('--lower', action="store_true", dest="lower", default=False,
                      help='Lower case the text: default=%default')
    parser.add_option('--lemmatize', action="store_true", dest="lemmatize", default=False,
                      help='Use lemmas instead of words: default=%default')
    parser.add_option('-n', dest='ngrams', default=2,
                      help='Max degree of word n-grams: default=%default')
    parser.add_option('--fast', action="store_true", dest="fast", default=False,
                      help='Only do things that are fast (i.e. only splitting, no parsing, no lemmas): default=%default')
    parser.add_option('-d', dest='display', default=1000,
                      help='Display progress every X items: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]

    suffix = options.suffix
    lower = options.lower
    lemmatize = options.lemmatize
    fast = options.fast
    ngrams = int(options.ngrams)
    display = int(options.display)

    preprocess_words(project_dir, subset, ngrams=ngrams, lower=lower, lemmatize=lemmatize, fast=fast, display=display, suffix=suffix)


def preprocess_words(project_dir, subset, ngrams=2, lower=False, lemmatize=False, fast=False, display=1000, suffix=''):

    print("Reading data")
    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')

    data = fh.read_json(datafile)
    keys = list(data.keys())
    keys.sort()

    if not fast:
        print("Loading spacy")
        parser = English()

    items = []
    unigrams = {}
    bigrams = {}
    ngram_dicts = {}
    for n in range(2, ngrams+1):
        ngram_dicts[n] = {}

    print("Parsing text")

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

        if not fast:
            # parse the text with spaCy
            parse = parser(text)

            if lemmatize:
                percept = get_lemma
            else:
                percept = get_word

            unigrams[name] = extract_unigram_feature(parse, percept)
            for n in range(2, ngrams+1):
                ngram_dicts[n][name] = extract_ngram_feature(parse, percept, n=n)

        else:
            # for fast processing:
            # remove punctuation
            clean_text = re.sub('[.,!?:;"`\']', '', text)
            parse = clean_text.split()
            percept = get_token

            unigrams[name] = extract_unigram_feature(parse, percept)
            if ngrams > 1:
                counter = Counter()
                counter.update([percept(parse[i]) + '_' + percept(parse[i+1]) for i in range(len(parse)-1)])
                bigrams[name] = dict(counter)


    #print("Creating word features")
    #word_feature = features.create_from_dict_of_counts('unigrams', unigrams)
    #word_feature.save_feature(dirs.dir_features(project_dir, subset))

    feature_dicts = {'unigrams': unigrams}
    for n in range(2, ngrams+1):
        key = 'n' + str(n) + 'grams'
        feature_dicts[key] = ngram_dicts[n]

    print("Creating features")
    for k, v in feature_dicts.items():
        feature = features.create_from_dict_of_counts(k + suffix, v)
        feature.save_feature(dirs.dir_features(project_dir, subset))


def get_word(token):
    """Get word from spaCy"""
    #  get word and remove whitespace
    word = re.sub('\s', '', token.orth_)
    return word


def get_lemma(token):
    """Get token from spaCy"""
    # get lemma and remove whitespace
    lemma = re.sub('\s', '', token.lemma_)
    return lemma


def get_token(token):
    """Identity function for compatibility"""
    return token


def extract_unigram_feature(parse, feature_function):
    counter = Counter()
    counter.update([feature_function(token) for token in parse if len(feature_function(token)) > 0])
    return dict(counter)


def extract_bigram_feature(parse, percept):
    counter = Counter()
    for sent in parse.sents:
        if len(sent) > 1:
            for i in range(len(sent) - 1):
                percept1 = percept(sent[i])
                percept2 = percept(sent[i+1])
                if len(percept1) > 0 and len(percept2) > 0:
                    if re.match('[a-zA-Z0-9]', percept1) is not None and re.match('[a-zA-Z0-9]', percept2) is not None:
                        counter.update([percept1 + '_' + percept2])

    return dict(counter)


def extract_ngram_feature(parse, percept, n=3):
    counter = Counter()
    for sent in parse.sents:
        if len(sent) > 1:
            for i in range(len(sent) - (n-1)):
                percepts = [percept(sent[j]) for j in range(i, i+n)]
                if np.prod([len(p) for p in percepts]) > 0:
                    if np.prod([re.match(r'[a-zA-Z0-9]', p) is not None for p in percepts]):
                        counter.update(['_'.join(percepts)])

    return dict(counter)


if __name__ == '__main__':
    main()
