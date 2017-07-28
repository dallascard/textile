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
    parser.add_option('--strip_html', action="store_true", dest="strip_html", default=False,
                      help='Strip out HTML tags: default=%default')
    parser.add_option('--fast', action="store_true", dest="fast", default=False,
                      help='Only do things that are fast (i.e. only splitting, no parsing): default=%default')
    parser.add_option('--word2vec_file', dest='word2vec_file', default=None,
                      help='Location of word2vec.bin file for word vector document features: default=%default')
    parser.add_option('-d', dest='display', default=100,
                      help='Display progress every X items: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')
    lower = options.lower
    strip_html = options.strip_html
    fast = options.fast
    word2vec_file = options.word2vec_file
    display = int(options.display)

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())

    if not fast:
        print("Loading spacy")
        parser = English()

    items = []

    vectors = None
    vector_dim = 300
    if word2vec_file is not None:
        # load pre-trained word vectors
        print("Loading pre-trained word vectors")
        vectors = gensim.models.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

    print("Parsing texts")
    keys = list(data.keys())
    keys.sort()

    if word2vec_file is not None:
        mean_word_vectors = {}
        weighted_word_vectors = {}
        alpha = 10e-4

        print("Determining word weights for word vectors")
        n_items, n_terms = word_feature.shape
        binarized = preprocessing.binarize(word_feature.get_counts())
        col_sums = np.array(binarized.sum(0))
        doc_freqs = np.reshape(col_sums / float(n_items), (n_terms, ))

        word_frequencies = {w: doc_freqs[i] for i, w in enumerate(word_feature.terms)}

        print("Extracting word vectors")
        for k_i, key in enumerate(keys):
            if k_i % display == 0 and k_i > 0:
                print(k_i)

            item = data[key]

            if 'name' in item:
                name = item['name']
            else:
                name = str(key)

            text = item['text']

            if lower:
                text = text.lower()

            if strip_html:
                text = normalize_text.strip_html(text)

            text = re.sub('_', '-', text)

            # parse the text with spaCy
            if not fast:
                parse = parser(text)

            else:
                clean_text = re.sub('[.,!?:;"`\']', '', text)
                # split on whitespace
                unigrams = clean_text.split()



            n_words = 0
            mean_word_vector = np.zeros(vector_dim)
            weighted_word_vector = np.zeros(vector_dim)

            for word, count in words[name].items():
                if word in vectors:
                    mean_word_vector += vectors[word] * count
                    weighted_word_vector += vectors[word] * count * (alpha / (alpha + word_frequencies[word]))
                    n_words += count

            if n_words > 0:
                mean_word_vector /= float(n_words)
                weighted_word_vector /= float(n_words)

            if np.isnan(mean_word_vector).any():
                sys.exit("NaN encountered in mean word vector")

            mean_word_vectors[name] = mean_word_vector
            weighted_word_vectors[name] = weighted_word_vector

        print("Creating word2vec feature")
        word2vec_feature = features.create_from_dict_of_vectors('word2vec', mean_word_vectors)
        print("Saving to file")
        word2vec_feature.save_feature(dirs.dir_features(project_dir, subset))

        print("Creating weigthed word2vec feature")
        weighted_word2vec_feature = features.create_from_dict_of_vectors('weightedWord2vec', weighted_word_vectors)
        print("Saving to file")
        weighted_word2vec_feature.save_feature(dirs.dir_features(project_dir, subset))

    print("Saving labels")
    label_set = list(label_set)
    try:
        if np.all([label.isdigit for label in label_set]):
            label_index = {label: int(label) for label in label_set}
        else:
            label_set.sort()
            label_index = {label: i for i, label in enumerate(label_set)}
    except ValueError:    # for sting labels
        label_set.sort()
        label_index = {label: i for i, label in enumerate(label_set)}
    except AttributeError:  # for float labels
        label_set.sort()
        label_index = {label: i for i, label in enumerate(label_set)}

    int_labels = {k: [label_index[i] for i in item_labels] for k, item_labels in labels.items()}

    output_dir = dirs.dir_labels(project_dir, subset)
    fh.makedirs(output_dir)
    #int_labels_df.to_csv(os.path.join(output_dir, label_name + '.csv'))
    fh.write_to_json(label_index, os.path.join(output_dir, label_name + '_index.json'))
    fh.write_to_json(int_labels, os.path.join(output_dir, label_name + '.json'))

    print("Saving metadata")
    output_dir = dirs.dir_subset(project_dir, subset)
    metadata.to_csv(os.path.join(output_dir, 'metadata.csv'))


def get_word(token):
    #  get word and remove whitespace
    word = re.sub('\s', '', token.orth_)
    return word


def get_lemma(token):
    # get lemma and remove whitespace
    lemma = re.sub('\s', '', token.lemma_)
    return lemma


def extract_unigram_feature(parse, feature_function):
    counter = Counter()
    counter.update([feature_function(token) for token in parse])
    return dict(counter)


def extract_bigram_feature(parse, percept):
    counter = Counter()
    for sent in parse.sents:
        if len(sent) > 1:
            counter.update([percept(sent[i]) + '_' + percept(sent[i+1]) for i in range(len(sent)-1)])
    return dict(counter)


if __name__ == '__main__':
    main()
