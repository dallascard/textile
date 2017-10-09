import os
import sys
import json

import numpy as np
from scipy import sparse
from sklearn import preprocessing

from ..util import file_handling as fh
from ..util.misc import printv


def parse_feature_string(feature_string):
    parts = feature_string.split(',')
    name = parts[0]
    kwargs = {}
    for p in parts[1:]:
        parts2 = p.split('=')
        kwargs[parts2[0]] = parts2[1]
    return FeatureDef(name, **kwargs)


# TODO: figure out how to manage the relationship between feature_def and feature...
class FeatureDef:

    def __init__(self, name, min_df=0, max_fp=1.0, transform=None, alpha=None):
        self.name = name
        self.min_df = int(min_df)    # minimum document frequency for inclusion
        self.max_fp = float(max_fp)  # maximum frequency (percent) for inclusion
        self.transform = transform   # transform [binarize|normalize|tf-idf|doc2vec]
        self.alpha = alpha           # smoothing for doc2vec transform

    def __str__(self):
        data = json.dumps({'name': self.name, 'min_df': self.min_df, 'max_fp': self.max_fp, 'transform': self.transform, 'alpha': self.alpha})
        return data


class Feature:
    """
    Data structure for a feature matrix, represented as a matrix of (sparse by default) counts, along with 
    lists of names for the rows (items) and columns (names). The idea is that we can easily combine multiple
    features for a classifier and apply transformation such as tf-idf or limit the vocabulary.
    """

    def __init__(self, name, items, terms, counts, col_names=None):
        self.name = name       # name of feature
        assert type(items) == list
        self.items = items     # list of items
        assert type(terms) == list
        self.terms = terms     # list of terms (feature names)
        if sparse.issparse(counts):
            self.sparse = True
            assert sparse.isspmatrix_csc(counts)
        else:
            self.sparse = False
        self.counts = counts
        # add an additional member to be usd primarily for the case of vector transformations
        if col_names is None:
            self.col_names = terms[:]
        else:
            self.col_names = col_names

    def get_name(self):
        return self.name

    def get_items(self):
        return self.items

    def get_terms(self):
        return self.terms

    def get_col_names(self):
        return self.col_names

    def get_counts(self):
        return self.counts

    def get_shape(self):
        return self.counts.shape

    def get_row_in_index_format(self, row):
        row = self.counts[row]
        nonzero_indices = row.nonzero()[1]
        return ' '.join([str(i+1) + ':' + str(row[0, i]) for i in nonzero_indices])

    def save_feature(self, output_dir):
        fh.makedirs(output_dir)
        output_filename = os.path.join(output_dir, self.name + '.npz')
        assert self.sparse
        fh.save_sparse(self.counts, output_filename)
        # save index and column names separately for human readability
        fh.write_to_json({'items': self.items, 'terms': self.terms, 'col_names': self.col_names}, os.path.join(output_dir, self.name + '.json'), sort_keys=False)  # Feature.save: fh.write_to_json()

    def transform(self, transform, idf=None, verbose=False, word_vectors_prefix=None, alpha=None):
        """
        Apply a transform to the counts for this feature according to the transform parameter
        :param transform: type of transformation (of the type Transforms above)
        :param idf: if given, use these idf values for the tf-idf transform; if not, calculate internally
        :return: None
        """
        if transform is not None:
            printv("Transforming %s by %s" % (self.name, transform), verbose)
            assert sparse.isspmatrix_csc(self.counts)
            if transform == 'binarize':
                self.counts = preprocessing.binarize(self.counts)        # this should still be sparse
            elif transform == 'normalize':
                row_sums = self.counts.sum(1)
                inv_row_sum = 1.0 / np.maximum(1.0, row_sums)
                inv_row_sum_sparse = sparse.csc_matrix(inv_row_sum)
                self.counts = self.counts.multiply(inv_row_sum_sparse)   # this should still be sparse
            elif transform == 'tfidf':
                if idf is None:
                    idf = self.compute_idf()
                idf_sparse = sparse.csc_matrix(idf)
                self.counts = self.counts.multiply(idf_sparse)
            elif transform == 'doc2vec':
                assert word_vectors_prefix is not None
                printv("Loading word vectors")
                word_vectors = fh.load_dense(word_vectors_prefix + '.npz')
                word_vector_terms = fh.read_json(word_vectors_prefix + '.json')
                # restrict the terms to those found in the word vectors
                self.set_terms(word_vector_terms)
                self.counts = self.counts.tocsr()
                n_items = len(self.items)
                dv, dh = word_vectors.shape
                if alpha is not None:
                    p_w = self.compute_p_w()
                    word_weights = float(alpha) / np.array(float(alpha) + p_w, dtype=float)
                    word_vectors = word_weights.reshape(dv, 1) * word_vectors
                doc_vectors = np.zeros([n_items, dh])
                printv("Converting documents to doc_vectors")
                for i in range(n_items):
                    doc_vectors[i, :] = np.dot(self.counts[i, :].todense(), word_vectors)
                self.counts = doc_vectors  # this will no longer be sparse
                self.sparse = False
                self.col_names = [str(i) for i in range(dh)]
            else:
                sys.exit('Tranform %s not recognized' % transform)

    def compute_idf(self):
        """
        compute the inverse document frequency of a corpus
        This might be useful to obtain an idf measure from a training set and then pass it to a test set
        :return: a vector of idf values
        """
        n_items, n_terms = self.counts.shape
        binarized = preprocessing.binarize(self.counts)
        # compute the number of documents containing each term, and add 1.0 to avoid dividing by zero
        col_sums = binarized.sum(0) + 1.0
        idf = np.log(n_items / col_sums)
        return idf

    def compute_p_w(self):
        """
        Compute the approximate probability of each term
        :return: a vector of p(w) values
        """
        col_sums = np.array(self.counts.sum(axis=0), dtype=float)
        return col_sums / np.sum(col_sums)


    # TODO: keep these values internal? or just give a feature_def as an argument?
    # TODO: replace max_fp with something that drops the most common x% of terms
    def threshold(self, min_df=0, max_fp=1.0, verbose=False):
        assert sparse.isspmatrix_csc(self.counts)
        if min_df > 0:
            printv("Thresholding %s by min_df=%d" % (self.name, min_df), verbose)
            col_sums = self.counts.sum(axis=0)
            col_sums = np.array(col_sums)[0]  # flatten col_sums to 1 dimension
            indices = [i for i, v in enumerate(col_sums) if v >= min_df]
            self.counts = self.counts[:, indices]
            self.terms = [self.terms[i] for i in indices]
            self.col_names = self.terms[:]

        if max_fp < 1.0:
            printv("Thresholding %s by max_fp=%0.3f" % (self.name, max_fp), verbose)
            n_items, n_terms = self.get_shape()
            binarized = preprocessing.binarize(self.counts)
            col_sums = binarized.sum(axis=0)
            col_sums = np.array(col_sums)[0]  # flatten col_sums to 1 dimension
            col_freq = col_sums / float(n_items)
            indices = [i for i, v in enumerate(col_freq) if v <= max_fp]
            self.counts = self.counts[:, indices]
            self.terms = [self.terms[i] for i in indices]
            self.col_names = self.terms[:]

    def set_terms(self, terms):
        """
        Force self.counts to reflect the membership and order of the list 'terms', by selecting columns
                and inserting columns of zeros as necessary
        """
        n_items = len(self.items)
        n_terms = len(self.terms)
        zeros_col = sparse.csc_matrix(np.zeros([n_items, 1]))
        temp = sparse.hstack([self.counts, zeros_col])
        terms_index = dict(zip(self.terms + ['__zero_col__'], range(n_terms+1)))
        indices = [terms_index[t] if t in terms_index else n_terms for t in terms]
        self.counts = temp[:, indices]
        self.terms = terms


def create_from_counts(name, items, terms, counts):
    assert sparse.isspmatrix(counts)
    n_items, n_terms = counts.shape
    assert n_items == len(items)
    assert n_terms == len(terms)
    return Feature(name, items, terms, counts.tocsc())


def create_from_dense_array(name, items, terms, array):
    assert type(array) == np.array
    n_items, n_terms = array.shape
    assert n_items == len(items)
    assert n_terms == len(terms)
    return Feature(name, items, terms, array)


def create_from_dict_of_counts(name, dict_of_counters):
    keys = list(dict_of_counters.keys())
    keys.sort()
    n_items = len(keys)

    vocab = set()
    for key in keys:
        vocab.update(list(dict_of_counters[key].keys()))

    terms = list(vocab)
    terms.sort()
    n_terms = len(terms)
    term_index = dict(zip(terms, range(n_terms)))

    counts = sparse.lil_matrix((n_items, n_terms))

    print("Building matrix with %d items and %d unique terms" % (n_items, n_terms))

    for item_i, item in enumerate(keys):
        item_counts = list(dict_of_counters[item].values())
        item_terms = list(dict_of_counters[item].keys())
        indices = [term_index[t] for t in item_terms]
        counts[item_i, indices] = item_counts

    return Feature(name, keys, terms, counts.tocsc())


def load_from_file(input_dir, basename):
    metadata = fh.read_json(os.path.join(input_dir, basename + '.json'))
    counts = fh.load_sparse(os.path.join(input_dir, basename + '.npz'))    # sparse.csc_matrix
    if 'col_names' in metadata:
        col_names = metadata['col_names']
        return Feature(basename, metadata['items'], metadata['terms'], counts, col_names=col_names)
    else:
        return Feature(basename, metadata['items'], metadata['terms'], counts)


def create_from_feature(feature, indices):
    name = feature.get_name()
    items = feature.get_items()
    terms = feature.get_terms()
    counts = feature.get_counts()
    col_names = feature.get_col_names()
    return Feature(name, [items[i] for i in indices], terms, counts.tocsr()[indices, :].tocsc(), col_names=col_names)


def create_from_dict_of_vectors(name, dict_of_vectors):
    keys = list(dict_of_vectors.keys())
    keys.sort()
    n_items = len(keys)

    vector_dim = len(dict_of_vectors[keys[0]])
    terms = ['d' + str(i) for i in range(vector_dim)]
    n_terms = len(terms)
    term_index = dict(zip(terms, range(n_terms)))

    counts = np.zeros((n_items, n_terms))

    print("Building matrix with %d items and %d columns" % (n_items, n_terms))

    for item_i, item in enumerate(keys):
        item_vector = dict_of_vectors[item]
        counts[item_i, :] = item_vector

    return Feature(name, keys, terms, sparse.csc_matrix(counts))


def concatenate(features):
    items = None
    terms = []
    counts = []
    all_sparse = True
    # check if all the features are sparse
    for f_i, feature in enumerate(features):
        if not feature.sparse:
            all_sparse = False

    for f_i, feature in enumerate(features):
        # make sure the list of items is the same for all features
        if f_i == 0:
            items = feature.items
        else:
            assert items == feature.items
        terms.extend(feature.terms)    # concatenate the lists of terms
        if all_sparse:
            counts.append(feature.counts)  # store a list of counts
        else:
            if feature.sparse:
                counts.append(feature.counts.todense())
            else:
                counts.append(feature.counts)

    # create a new feature from concatenation of counts
    if all_sparse:
        return Feature('concat', items, terms, sparse.hstack(counts, format='csc'))
    else:
        return Feature('concat', items, terms, np.hstack(counts))


def concatenate_rows(features):
    items = []
    terms = []
    counts = []
    name = ''
    all_sparse = True
    # check if all the features are sparse
    for f_i, feature in enumerate(features):
        if not feature.sparse:
            all_sparse = False

    for f_i, feature in enumerate(features):
        # make sure the list of items is the same for all features
        if f_i == 0:
            name = feature.name
            terms = feature.terms
        else:
            assert name == feature.name
            assert terms == feature.terms
        items.extend(feature.items)
        if all_sparse:
            counts.append(feature.counts)  # store a list of counts
        else:
            if feature.sparse:
                counts.append(feature.counts.todense())
            else:
                counts.append(feature.counts)

    # create a new feature from concatenation of counts
    if all_sparse:
        return Feature(name, items, terms, sparse.vstack(counts, format='csc'))
    else:
        return Feature(name, items, terms, np.vstack(counts))


def get_feature_signature(feature_def, feature):
    signature = {}
    signature['name'] = feature_def.name
    signature['min_df'] = feature_def.min_df
    signature['max_fp'] = feature_def.max_fp
    signature['transform'] = feature_def.transform
    signature['terms'] = feature.get_terms()
    signature['col_names'] = feature.get_col_names()
    if feature_def.transform == 'tfidf':
        signature['idf'] = feature.compute_idf().tolist()
    elif feature_def.transform == 'doc2vec':
        signature['p_w'] = feature.compute_p_w().tolist()
    return signature


def load_and_process_features_for_training(features_dir, feature_defs, items_to_use, verbose=False):

    feature_list = []
    feature_signatures = []
    for feature_def in feature_defs:
        printv(feature_def, verbose)
        name = feature_def.name
        feature = load_from_file(input_dir=features_dir, basename=name)
        # take a subset of the rows, if requested
        printv("Initial shape = (%d, %d)" % feature.get_shape(), verbose)
        feature_items = feature.get_items()
        feature_item_index = dict(zip(feature_items, range(len(feature_items))))
        indices_to_use = [feature_item_index[i] for i in items_to_use if i in feature_item_index]
        if indices_to_use is not None:
            printv("Taking subset of items", verbose)
            feature = create_from_feature(feature, indices_to_use)
            printv("New shape = (%d, %d)" % feature.get_shape(), verbose)
        feature.threshold(feature_def.min_df)
        if feature_def.transform == 'doc2vec':
            word_vectors_prefix = os.path.join(features_dir, name + '_vecs')
        else:
            word_vectors_prefix = None
        feature.transform(feature_def.transform, word_vectors_prefix=word_vectors_prefix, alpha=feature_def.alpha)
        printv("Final shape = (%d, %d)" % feature.get_shape(), verbose)
        feature_list.append(feature)

        feature_signature = get_feature_signature(feature_def, feature)
        # save the location of the word vectors from training... (need a better solution for this eventually)
        feature_signature['word_vectors_prefix'] = word_vectors_prefix
        feature_signatures.append(feature_signature)

    features_concat = concatenate(feature_list)
    return features_concat, feature_signatures
