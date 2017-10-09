import os
from optparse import OptionParser

import numpy as np

from ..preprocessing import features

from ..util import dirs
from ..util import file_handling as fh
from ..util.misc import printv

def main():
    usage = "%prog target_feature.json background_feature.json"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n', default=100,
                      help='Number of words: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    target_feature_file = args[0]
    bkgrnd_feature_file = args[1]

    n = int(options.n)

    load_and_select_features(target_feature_file, bkgrnd_feature_file, n)


def load_and_select_features(target_json_file, bkgrnd_json_file, n=100):

    target_dir, target_filename = os.path.split(target_json_file)
    target_basename, _ = os.path.splitext(target_filename)

    bkgrnd_dir, bkgrnd_filename = os.path.split(bkgrnd_json_file)
    bkgrnd_basename, _ = os.path.splitext(bkgrnd_filename)

    target_feature = features.load_from_file(target_dir, target_basename)
    bkgrnd_feature = features.load_from_file(bkgrnd_dir, bkgrnd_basename)

    vocab, scores = select_features(target_feature, bkgrnd_feature, n=n)
    for i in range(len(vocab)):
        print(vocab[i], scores[i])


def select_features(feature, background_feature, n=100):
    """
    Use the method from "Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of
    Political Conflict" by Monroe, Colaresi and Quinn for feature selection.
    :param feature: a Feature object for the corpus of interest
    :param background_feature: a Feature object corresponding to the same feature in a background corpus
    :return: 
    """

    target_vocab = feature.get_terms()
    target_sum = np.array(feature.get_counts().sum(axis=0)).reshape(len(target_vocab))
    bkgrnd_vocab = background_feature.get_terms()
    bkgrnd_sum = np.array(background_feature.get_counts().sum(axis=0)).reshape(len(bkgrnd_vocab))

    print(len(target_vocab))
    print(len(bkgrnd_vocab))

    # construct the combined vocabulary
    full_vocab = list(set(target_vocab).union(set(bkgrnd_vocab)))
    full_vocab.sort()
    n_words = len(full_vocab)
    vocab_index = dict(zip(full_vocab, range(n_words)))

    # convert feature counts to arrays matching this vocbaulary
    target_counts = np.zeros(n_words)
    bkgrnd_counts = np.zeros(n_words)
    for word_i, word in enumerate(target_vocab):
        target_counts[vocab_index[word]] = target_sum[word_i]
    for word_i, word in enumerate(bkgrnd_vocab):
        bkgrnd_counts[vocab_index[word]] = bkgrnd_sum[word_i]

    alphas = get_informative_alpha(target_counts, bkgrnd_counts)
    word_scores = log_odds_normalized_diff(target_counts, bkgrnd_counts, alphas)
    order = list(np.argsort(np.abs(word_scores)))
    order.reverse()
    vocab = [full_vocab[i] for i in order[:n]]
    scores = [word_scores[i] for i in order[:n]]
    return vocab, scores


def get_informative_alpha(first_counts, second_counts, smoothing=1000.0):
    total_words = first_counts.sum() + second_counts.sum()
    alpha = (first_counts + second_counts) / float(total_words) * smoothing
    return alpha


def log_odds_normalized_diff(first_counts, second_counts, alphas):
    first_total = first_counts.sum()
    second_total = second_counts.sum()
    alpha_total = alphas.sum()
    first_top = first_counts + alphas
    first_bottom = first_total + alpha_total - first_counts - alphas
    second_top = second_counts + alphas
    second_bottom = second_total + alpha_total - second_counts - alphas
    delta = np.log(first_top) - np.log(first_bottom) - np.log(second_top) + np.log(second_bottom)
    variance = 1. / first_top + 1. / first_bottom + 1. / second_top + 1. / second_bottom
    word_scores = delta / np.sqrt(variance)
    return word_scores


def load_from_config_files(project, target_subset, background_subset, config_file, items_to_use=None, n=100, verbose=True):

    target_features_dir = dirs.dir_features(project, target_subset)
    background_features_dir = dirs.dir_features(project, background_subset)

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    target_feature, _ = features.load_and_process_features_for_training(target_features_dir, feature_defs, items_to_use, verbose=False)

    background_feature, _ = features.load_and_process_features_for_training(background_features_dir, feature_defs, items_to_use, verbose=False)

    return select_features(target_feature, background_feature, n=n)


if __name__ == '__main__':
    main()
