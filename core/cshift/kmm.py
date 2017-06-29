import math
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from cvxopt import matrix, solvers, spmatrix

from ..util import dirs
from ..util import file_handling as fh
from ..preprocessing import features


def main():
    usage = "%prog project source_subset target_subset config.json output_filename"
    parser = OptionParser(usage=usage)
    parser.add_option('-B', dest='B', default=10.0,
                      help='Upper bound on weights: default=%default')
    parser.add_option('-e', dest='eps', default=None,
                      help='epsilon parameter [None=B/sqrt(n)]: default=%default')
    parser.add_option('--sparse', action="store_true", dest="sparse", default=False,
                      help='Treat feature matrices as sparse: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    source_subset = args[1]
    target_subset = args[2]
    config_file = args[3]
    output_filename = args[4]

    B = float(options.B)
    is_sparse = options.sparse
    eps = options.eps
    if eps is not None:
        eps = float(eps)

    weights = compute_weights(project, source_subset, target_subset, config_file, B, eps, is_sparse)
    weights.to_csv(output_filename)


def compute_weights(project, source_subset, target_subset, config_file, B=10, eps=None, is_sparse=True):

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    source_dir = dirs.dir_features(project, source_subset)
    target_dir = dirs.dir_features(project, target_subset)

    # for each feature, combine source and target to determine which terms to keep, then produce each separately
    print("loading features")
    source_feature_list = []
    target_feature_list = []
    for feature_def in feature_defs:
        print(feature_def)
        name = feature_def.name
        source_feature = features.load_from_file(input_dir=source_dir, basename=name)
        target_feature = features.load_from_file(input_dir=target_dir, basename=name)
        print("Source:", source_feature.get_shape())
        print("Target:", target_feature.get_shape())
        common_terms = list(set(source_feature.get_terms()).intersection(set(target_feature.get_terms())))
        source_feature.set_terms(common_terms)
        target_feature.set_terms(common_terms)
        print("Concatenating source and target")
        common_feature = features.concatenate_rows([source_feature, target_feature])
        print("Common:", common_feature.get_shape())
        common_feature.threshold(feature_def.min_df)
        common_feature.transform(feature_def.transform)
        idf = None
        if feature_def.transform == 'tfidf':
            idf = common_feature.compute_idf()
        common_feature.threshold(feature_def.min_df)
        source_feature.set_terms(common_feature.get_terms())
        target_feature.set_terms(common_feature.get_terms())
        source_feature.transform(feature_def.transform, idf=idf)
        target_feature.transform(feature_def.transform, idf=idf)
        print("Source:", source_feature.get_shape())
        print("Target:", target_feature.get_shape())
        source_feature_list.append(source_feature)
        target_feature_list.append(target_feature)

    source_features_concat = features.concatenate(source_feature_list)
    target_features_concat = features.concatenate(target_feature_list)
    col_names = source_features_concat.terms

    # code for conversion to a CVX sparse matrix:
    #coo = A.tocoo()
    #SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())

    if is_sparse:
        source_X = source_features_concat.get_counts()
        target_X = target_features_concat.get_counts()
    else:
        source_X = source_features_concat.get_counts().todense()
        target_X = target_features_concat.get_counts().todense()

    weights = do_kernel_mean_matching(source_X, target_X, kern='lin', B=B, eps=eps, is_sparse=is_sparse)

    print(weights.shape)
    print("min mean max:")
    print(np.min(weights), np.mean(weights), np.max(weights))
    return pd.DataFrame(weights, index=source_features_concat.get_items(), columns='weight')


def do_kernel_mean_matching(source_X, target_X, kern='lin', B=1.0, eps=None, is_sparse=False):
    n_source_items, p = source_X.shape
    n_target_items, _ = target_X.shape
    if eps == None:
        eps = B/math.sqrt(n_source_items)
    if kern == 'lin':
        if is_sparse:
            assert sparse.isspmatrix(source_X)
            assert sparse.isspmatrix(target_X)
            print("Computing K")
            dense_source_X = source_X.todense()
            dense_target_X = target_X.todense()
            K = sparse.csc_matrix(np.dot(dense_source_X, dense_source_X.T))
            #K = source_X.dot(source_X.T)
            print("Computing kappa")
            kappa = np.dot(dense_source_X, dense_target_X.T).sum(axis=1) * float(n_source_items) / float(n_target_items)
        else:
            K = np.dot(source_X, source_X.T)
            kappa = np.sum(np.dot(source_X, target_X.T), axis=1) * float(n_source_items) / float(n_target_items)
    elif kern == 'rbf':
        K = compute_rbf(source_X, source_X)
        kappa = np.sum(compute_rbf(source_X, target_X), axis=1) * float(n_target_items) / float(n_source_items)
    else:
        raise ValueError('unknown kernel')

    if is_sparse:
        print("Making spmatrices")
        K = make_spmatrix_from_sparse(K)
    else:
        K = matrix(K)
    kappa = matrix(kappa)

    print("Creating constraint matrices")
    # will enforce G \cdot \beta <= h
    # first two are for: | -n_source_items + \sum \beta | <= n_source_items * eps
    # second two are for : \beta \in [0, B]
    G = np.r_[np.ones((1, n_source_items)), -np.ones((1, n_source_items)), np.eye(n_source_items), -np.eye(n_source_items)]
    h = np.r_[n_source_items * (1 + eps), n_source_items * (eps - 1), B * np.ones((n_source_items,)), np.zeros((n_source_items,))]

    if is_sparse:
        G = make_spmatrix_from_sparse(sparse.coo_matrix(G))
        h = make_spmatrix_from_sparse(sparse.coo_matrix(h))
    else:
        G = matrix(G)
        h = matrix(h)

    print("Calling CVX")
    sol = solvers.qp(K, -kappa, G, h)
    beta = np.array(sol['x'])
    return beta


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
    return K


def make_spmatrix_from_sparse(X):
    if sparse.isspmatrix_coo(X):
        X_coo = X
    else:
        X_coo = X.tocoo()

    X_spmatrix = spmatrix(X_coo.data.tolist(), X_coo.row.tolist(), X_coo.col.tolist(), size=X.shape)
    return X_spmatrix


if __name__ == '__main__':
    main()
