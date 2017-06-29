import math
from optparse import OptionParser

import numpy as np
from cvxopt import matrix, solvers

from ..util import dirs
from ..util import file_handling as fh
from ..preprocessing import features

def main():
    usage = "%prog project source_subset target_subset config.json"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    source_subset = args[1]
    target_subset = args[2]
    config_file = args[3]

    compute_weights(project, source_subset, target_subset, config_file)


def compute_weights(project, source_subset, target_subset, config_file):

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
        print("Source:", source_feature.shape)
        print("Target:", target_feature.shape)
        common_terms = list(set(source_feature.get_terms()).intersection(set(target_feature.get_terms())))
        source_feature.set_terms(common_terms)
        target_feature.set_terms(common_terms)
        print("Concatenating source and target")
        common_feature = features.concatenate_rows([source_feature, target_feature])
        print("Common:", common_feature.shape)
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
        print("Source:", source_feature.shape)
        print("Target:", target_feature.shape)
        source_feature_list.append(source_feature)
        target_feature_list.append(target_feature)

    source_features_concat = features.concatenate(source_feature_list)
    target_features_concat = features.concatenate(target_feature_list)
    col_names = source_features_concat.terms

    source_X = source_features_concat.get_counts().tocsr()
    target_X = target_features_concat.get_counts().tocsr()
    coefs = do_kernel_mean_matching(source_X, target_X, kern='lin')

    print(coefs.shape)
    print(np.min(coefs), np.mean(coefs), np.max(coefs))


# an implementation of Kernel Mean Matching
def do_kernel_mean_matching(source_X, target_X, kern='lin', B=1.0, eps=None):
    n_source_items, p = source_X.shape
    n_target_items, _ = target_X.shape
    if eps == None:
        eps = B/math.sqrt(n_target_items)
    if kern == 'lin':
        K = np.dot(target_X, target_X.T)
        kappa = np.sum(np.dot(target_X, source_X.T) * float(n_target_items) / float(n_source_items), axis=1)
    elif kern == 'rbf':
        K = compute_rbf(target_X, target_X)
        kappa = np.sum(compute_rbf(target_X, source_X), axis=1) * float(n_target_items) / float(n_source_items)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, n_target_items)), -np.ones((1, n_target_items)), np.eye(n_target_items), -np.eye(n_target_items)])
    h = matrix(np.r_[n_target_items * (1+eps), n_target_items * (eps-1), B*np.ones((n_target_items,)), np.zeros((n_target_items,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i,:] = np.exp(-np.sum((vx-Z)**2, axis=1)/(2.0*sigma))
    return K



if __name__ == '__main__':
    main()
