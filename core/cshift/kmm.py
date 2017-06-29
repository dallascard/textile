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
    usage = "%prog project source_subset target_subset config.json output_filename.csv"
    parser = OptionParser(usage=usage)
    parser.add_option('-B', dest='B', default=10.0,
                      help='Upper bound on weights: default=%default')
    parser.add_option('-e', dest='eps', default=None,
                      help='epsilon parameter [None=B/sqrt(n)]: default=%default')
    parser.add_option('-k', dest='kernel', default='poly',
                      help='Kernel type [poly|rbf]: default=%default')
    parser.add_option('-b', dest='bandwidth', default=1.0,
                      help='Bandwidth of rbf kernel: default=%default')
    parser.add_option('-c', dest='offset', default=0.0,
                      help='Offset of polynomial kernel [0=homogeneous]: default=%default')
    parser.add_option('-d', dest='degree', default=1,
                      help='Degree of polynomial kernel [1=linear]: default=%default')
    #parser.add_option('--sparse', action="store_true", dest="sparse", default=False,
    #                  help='Treat feature matrices as sparse: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    source_subset = args[1]
    target_subset = args[2]
    config_file = args[3]
    output_filename = args[4]

    B = float(options.B)
    eps = options.eps
    if eps is not None:
        eps = float(eps)
    kernel = options.kernel
    bandwidth = float(options.bandwidth)
    degree = int(options.degree)
    offset = float(options.offset)

    weights = compute_weights(project, source_subset, target_subset, config_file, B, eps, kernel, bandwidth, offset, degree)
    weights.to_csv(output_filename)


def compute_weights(project, source_subset, target_subset, config_file, B=10.0, eps=None, kernel='poly', bandwidth=1.0, offset=0.0, degree=1):

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

    source_X = source_features_concat.get_counts().todense()
    target_X = target_features_concat.get_counts().todense()

    weights = do_kernel_mean_matching(source_X, target_X, B=B, eps=eps, kernel=kernel, bandwidth=bandwidth, offset=offset, degree=degree)

    print(weights.shape)
    print("min mean max:")
    print(np.min(weights), np.mean(weights), np.max(weights))
    return pd.DataFrame(weights, index=source_features_concat.get_items(), columns=['weight'])


def do_kernel_mean_matching(source_X, target_X, B=1.0, eps=None, kernel='poly', bandwidth=1.0, offset=0.0, degree=1):
    n_source_items, p = source_X.shape
    n_target_items, _ = target_X.shape
    if eps == None:
        eps = B/math.sqrt(n_source_items)
    if kernel == 'poly':
        K = compute_poly(source_X, source_X, degree, offset)
        kappa = np.sum(compute_poly(source_X, target_X, degree, offset), axis=1) * float(n_source_items) / float(n_target_items)
    elif kernel == 'rbf':
        K = compute_rbf(source_X, source_X, bandwidth)
        kappa = np.sum(compute_rbf(source_X, target_X, bandwidth), axis=1) * float(n_target_items) / float(n_source_items)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)

    print("Creating constraint matrices")
    # will enforce G \cdot \beta <= h
    # first two are for: | -n_source_items + \sum \beta | <= n_source_items * eps
    # second two are for : \beta \in [0, B]
    G = np.r_[np.ones((1, n_source_items)), -np.ones((1, n_source_items)), np.eye(n_source_items), -np.eye(n_source_items)]
    h = np.r_[n_source_items * (1 + eps), n_source_items * (eps - 1), B * np.ones((n_source_items,)), np.zeros((n_source_items,))]

    G = make_spmatrix_from_sparse(sparse.coo_matrix(G))
    h = matrix(h)

    print("Calling CVX")
    sol = solvers.qp(K, -kappa, G, h)
    beta = np.array(sol['x'])
    return beta


def compute_poly(X, Y, degree=1, offset=0.0):
    if degree == 1:
        K = np.dot(X, Y.T) + offset
    else:
        K = np.power(np.dot(X, Y.T) + offset, degree)
    return K


def compute_rbf(X, Y, bandwidth=1.0):
    K = np.zeros((X.shape[0], Y.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i,:] = np.exp(-bandwidth * np.sum((vx-Y)**2, axis=1))
    return K


def make_spmatrix_from_sparse(X):
    if sparse.isspmatrix_coo(X):
        X_coo = X
    else:
        X_coo = X.tocoo()

    print("Size=%d" % len(X_coo.data.tolist()))
    X_spmatrix = spmatrix(X_coo.data.tolist(), X_coo.row.tolist(), X_coo.col.tolist(), size=X.shape)
    return X_spmatrix


if __name__ == '__main__':
    main()
