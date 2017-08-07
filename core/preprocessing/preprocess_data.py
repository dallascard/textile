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
    parser.add_option('--label', dest='label', default='label',
                      help='Name of label (in data_file.json): default=%default')
    parser.add_option('--lower', action="store_true", dest="lower", default=False,
                      help='Lower case the text: default=%default')
    parser.add_option('-w', dest='wgrams', default=2,
                      help='Max degree of word n-grams [0, 1 or 2]: default=%default')
    parser.add_option('-c', dest='cgrams', default=0,
                      help='Max degree of character n-grams: default=%default')
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
    label_name = options.label
    fast = options.fast
    wgrams = int(options.wgrams)
    cgrams = int(options.cgrams)
    word2vec_file = options.word2vec_file
    display = int(options.display)

    print("Reading data")
    data = fh.read_json(datafile)
    keys = list(data.keys())

    # make a list of metadata fields (not text or label)
    fields = list(data[keys[0]].keys())
    print(fields)
    fields.remove('text')
    fields.remove(label_name)

    if not fast:
        print("Loading spacy")
        parser = English()

    items = []
    words = {}
    bigrams = {}
    chargrams = {}

    vectors = None
    vector_dim = 300
    if word2vec_file is not None:
        # load pre-trained word vectors
        print("Loading pre-trained word vectors")
        vectors = gensim.models.Word2Vec.load_word2vec_format(word2vec_file, binary=True)

    metadata = pd.DataFrame(columns=fields)
    labels = {}
    label_set = set()

    print("Parsing texts")
    keys = list(data.keys())
    keys.sort()

    unigram_vocab = set()
    bigram_vocab = set()
    chargram_vocab = set()

    for k_i, key in enumerate(keys):
        if k_i % display == 0 and k_i > 0:
            print(k_i)

        item = data[key]
        if type(item[label_name]) == dict:
            label_dict = item[label_name]
        else:
            label_dict = {item[label_name]: 1}
        label_set.update(label_dict.keys())

        text = item['text']

        if lower:
            text = text.lower()

        if strip_html:
            text = normalize_text.strip_html(text)

        if 'name' in item:
            name = item['name']
        else:
            name = str(key)
        items.append(name)

        metadata.loc[name] = [item[f] for f in fields]
        labels[name] = label_dict

        # extract the bigrams from each sentence
        #bigram_counter = Counter()
        #sents = parse.sents
        #for sent in sents:
        #    sent_bigrams = [sent[i].orth_ + '_' + sent[i+1].orth_ for i in range(len(sent)-1)]
        #    bigram_counter.update(sent_bigrams)

        #for token in parse:
        #    word = token.orth_
        #    shape = token.shape_
        #    lemma = token.lemma_
        #    pos = token.pos_
        #    tag = token.tag_
        #    dep = token.dep_
        #    head_token = token.head
        #    children_tokens = token.children
        #    ent_type = token.ent_type_
        #    prefix = token.prefix_
        #    suffix = token.suffix_

        # replace underscores with dashes to avoid confusion
        text = re.sub('_', '-', text)

        # parse the text with spaCy
        if not fast:
            parse = parser(text)

            if wgrams > 0:
                words[name] = extract_unigram_feature(parse, get_word)
            if wgrams > 1:
                bigrams[name] = extract_bigram_feature(parse, get_word)

            if cgrams > 0:
                letters = list(text)
                counter = Counter()
                for c in range(1, cgrams+1):
                    counter.update([''.join(letters[i:i+c]) for i in range(len(letters)-c+1)])
                chargrams[name] = dict(counter)

        else:
            # for fast processing:
            # remove punctuation
            clean_text = re.sub('[.,!?:;"`\']', '', text)
            # split on whitespace
            unigrams = clean_text.split()
            if wgrams > 0:
                unigram_vocab.update(unigrams)

            if wgrams > 1:
                bigram_vocab.update([unigrams[i] + '_' + unigrams[i+1] for i in range(len(unigrams)-1)])

            if cgrams > 0:
                letters = list(text)
                for c in range(1, cgrams+1):
                    chargram_vocab.update(set([''.join(letters[i:i+c]) for i in range(len(letters)-c+1)]))

    if not fast:
        print("Creating word features")
        if wgrams > 0:
            word_feature = features.create_from_dict_of_counts('words', words)
            word_feature.save_feature(dirs.dir_features(project_dir, subset))

        if wgrams > 1:
            bigram_feature = features.create_from_dict_of_counts('bigrams', bigrams)
            bigram_feature.save_feature(dirs.dir_features(project_dir, subset))

        if cgrams > 0:
            chargram_feature = features.create_from_dict_of_counts('chargrams', chargrams)
            chargram_feature.save_feature(dirs.dir_features(project_dir, subset))

    else:
        unigram_vocab = list(unigram_vocab)
        unigram_counts = sparse.csr_matrix([n_items, len(unigram_vocab)])

        for k_i, key in enumerate(keys):
            if k_i % display == 0 and k_i > 0:
                print(k_i)

                print("Building count matrices")
                # for fast processing:
                # remove punctuation
                clean_text = re.sub('[.,!?:;"`\']', '', text)
                # split on whitespace
                unigrams = clean_text.split()
                if wgrams > 0:
                    unigram_counter = Counter()
                    unigram_counter.update(unigrams)
                    words[name] = dict(unigram_counter)

                if wgrams > 1:
                    bigram_counter = Counter()
                    if len(unigrams) > 1:
                        bigram_counter.update([unigrams[i] + '_' + unigrams[i+1] for i in range(len(unigrams)-1)])
                    bigrams[name] = dict(bigram_counter)


    if word2vec_file is not None and wgrams > 0:
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
