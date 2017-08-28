import os
import gensim
from optparse import OptionParser

import numpy as np

from ..util import file_handling as fh
from ..util import dirs
from ..preprocessing import features


def main():
    """
    Just extract the word vectors required for a particular subset and store them in a matrix
    :return: 
    """
    usage = "%prog project_dir subset word2vec_file.bin"
    parser = OptionParser(usage=usage)
    parser.add_option('--ref', dest='ref', default='unigrams',
                      help='Reference feature definition: default=%default')
    parser.add_option('-s', dest='size', default=300,
                      help='Size of word vectors: default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    word2vec_file = args[2]

    ref = options.ref
    dh = int(options.size)

    preprocess_word_vectors(project_dir, subset, word2vec_file, ref, dh)


def preprocess_word_vectors(project_dir, subset, word2vec_file, ref, dh):

    print("Loading %s" % ref)
    unigrams = features.load_from_file(dirs.dir_features(project_dir, subset), ref)

    n_items, n_terms = unigrams.get_shape()
    terms = unigrams.get_terms()
    print("(%d, %d)" % (n_items, n_terms))

    print("Loading word vectors")
    vectors = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)

    # create the list of terms that have vectors
    vocab = [w for w in terms if w in vectors]
    dv = len(vocab)

    W = np.zeros([dv, dh])

    print("Extracting subset")
    for w_i, word in enumerate(vocab):
        if word in vectors:
            W[w_i, :] = vectors[word]

    print("Found vectors for %d words" % dv)

    output_prefix = os.path.join(dirs.dir_features(project_dir, subset), ref + '_vecs')
    print("Saving to %s" % output_prefix)
    output_filename = output_prefix + '.npz'
    fh.save_dense(W, output_filename)
    fh.write_to_json(vocab, output_prefix + '.json', sort_keys=False)


if __name__ == '__main__':
    main()
