import os
import re
import sys
import gensim
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn import preprocessing
from spacy.en import English

from ..util import file_handling as fh
from ..preprocessing import normalize_text, features
from ..util import dirs


def main():
    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    #parser.add_option('--reference_subset', dest='reference', default=None,
    #                  help='Reference subset (i.e. train) for ensuring full : default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')


    print("Reading data")
    data = fh.read_json(datafile)

    print("Loading spacy")
    parser = English()
    #tagger = phrasemachine.get_stdeng_nltk_tagger()

    labels = []
    names = []

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
            word = token.orth_
            shape = token.shape_
            lemma = token.lemma_
            pos = token.pos_
            tag = token.tag_
            dep = token.dep_
            head_token = token.head
            children_tokens = token.children
            ent_type = token.ent_type_
            prefix = token.prefix_
            suffix = token.suffix_


        labels.loc[name] = [label]


    print("Creating word features")
    word_feature = features.create_from_dict_of_counts('words', words)
    word_feature.save_feature(dirs.dir_features(project_dir, subset))

    bigram_feature = features.create_from_dict_of_counts('bigrams', bigrams)
    bigram_feature.save_feature(dirs.dir_features(project_dir, subset))


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
            if k_i % 100 == 0 and k_i > 0:
                print(k_i)

            item = data[key]

            if 'name' in item:
                name = item['name']
            else:
                name = str(key)

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
    label_set = list(set(labels[label_name]))
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

    label_list = list(labels[label_name])
    int_labels_df = pd.DataFrame([label_index[label] for label in label_list], index=labels.index, columns=[label_name], dtype=int)

    output_dir = dirs.dir_labels(project_dir, subset)
    fh.makedirs(output_dir)
    int_labels_df.to_csv(os.path.join(output_dir, label_name + '.csv'))
    fh.write_to_json(label_index, os.path.join(output_dir, label_name + '_index.json'))


def get_word(token, lower=False):
    #  get word and remove whitespace
    return re.sub('\s', '', token.orth_)


def get_lemma(token):
    # get lemma and remove whitespace
    return re.sub('\s', '', token.lemma_)


def extract_unigram_feature(parse, feature_function):
    counter = Counter()
    counter.update([feature_function(token) for token in parse])
    return counter


def extract_bigram_feature(parse, percept):
    counter = Counter()
    for sent in parse.sents:
        if len(sent) > 1:
            counter.update([percept(sent[i]) + '_' + percept(sent[i+1]) for i in range(len(sent)-1)])
    return counter


if __name__ == '__main__':
    main()
