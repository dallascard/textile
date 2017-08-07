import os
from optparse import OptionParser

import gensim
import numpy as np
import tensorflow as tf

from ..util import file_handling as fh
from ..models import evaluation, calibration
from ..preprocessing import features
from ..util import dirs


def main():
    usage = "%prog project_dir subset word2vec_file"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Label to use: default=%default')
    parser.add_option('--load', action="store_true", dest="load", default=False,
                      help='Load embeddings from last run: default=%default')


    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    word2vec_file = args[2]

    label = options.label
    load = options.load

    train(project_dir, subset, word2vec_file, label=label, load=load)


def train(project_dir, subset, word2vec_file, label='label', load=False):
    label_dir = dirs.dir_labels(project_dir, subset)
    labels_df = fh.read_csv_to_df(os.path.join(label_dir, label + '.csv'), index_col=0, header=0)

    features_dir = dirs.dir_features(project_dir, subset)

    feature = features.load_from_file(input_dir=features_dir, basename='unigrams')
    # take a subset of the rows, if requested
    print("Initial shape = (%d, %d)" % feature.get_shape())
    feature.threshold(10)
    feature.transform('binarize')
    print("Final shape = (%d, %d)" % feature.get_shape())

    vocab = feature.get_terms()
    dv = len(vocab)
    X = feature.get_counts().todense()
    word_counts = X.sum(axis=0)
    word_freqs = word_counts / float(word_counts.sum())
    a = 1e-3
    word_weights = np.array(a / (a + word_freqs))
    Y = labels_df.as_matrix()
    n_items, n_classes = Y.shape
    weights = np.ones(n_items)
    print(X.shape, Y.shape, weights.shape)
    X, Y, weights = expand_features_and_labels(X, Y, weights)
    print(X.shape, Y.shape, weights.shape)

    print(vocab[:10])
    print("Loading vectors")
    dh = 300

    if load:
        W = np.load('W.npz')['W']
        #W = None
    else:
        vectors = gensim.models.KeyedVectors.load_word2vec_format(word2vec_file, binary=True)
        W = np.zeros([dv, dh])
        vocab_index = dict(zip(vocab, range(dv)))
        for w_i, word in enumerate(vocab):
            if word in vectors:
                W[w_i, :] = vectors[word]

        np.savez('W.npz', W=W)

    print("W")
    print(np.min(W), np.max(W))

    W = W * word_weights.reshape((dv, 1))

    # define network
    x = tf.placeholder(tf.float32, shape=[None, dh])
    y = tf.placeholder(tf.int32, shape=[None, 2])
    sample_weights = tf.placeholder(tf.float32)
    #p_w = tf.placeholder((1, dv))

    w = weight_variable((dh, 2))
    #w2 = weight_variable((dh, 2))
    b = bias_variable((2, ))
    #b2 = bias_variable((2, ))

    #v = tf.multiply(a, 1.0/tf.add(a, word_freqs))
    #h = tf.reduce_sum(x, 1)
    s = tf.matmul(x, w) + b

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=s))

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy * sample_weights)
    pred = tf.argmax(s)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(100):
            counts_i = X[i, :]
            x_i = np.dot(counts_i.reshape((1, dv)), W)
            #print(x_i.shape)
            y_i = np.array(Y[i], dtype=np.int32).reshape((1, 2))
            w_i = weights[i]
            sess.run(train_step, feed_dict={x: x_i, y: y_i, sample_weights: w_i})
            score = sess.run(s, feed_dict={x: x_i, y: y_i, sample_weights: w_i})
            bias = sess.run(b, feed_dict={x: x_i, y: y_i, sample_weights: w_i})
            ws = sess.run(w, feed_dict={x: x_i, y: y_i, sample_weights: w_i})
            #h = sess.run(h, feed_dict={x: x_i, y: y_i})
            print(y_i[0], w_i, score[0], bias, np.min(ws), np.max(ws))

        test_X = W
        test_Y = np.zeros((dv, 2))
        test_weights = np.ones(dv)
        scores = sess.run(s, feed_dict={x: test_X, y: test_Y, sample_weights: test_weights})
        pos_scores = scores[:, 1]
        order = list(np.argsort(pos_scores))
        print("NEG")
        for i in range(20):
            print(vocab[order[i]])
        order.reverse()
        print("POS")
        for i in range(20):
            print(vocab[order[i]])


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def expand_features_and_labels(X, Y, weights):
    X_list = []
    Y_list = []
    weights_list = []
    n_items, n_classes = Y.shape
    for i in range(n_items):
        n_neg, n_pos = Y[i, :]
        for n in range(n_neg):
            X_list.append(np.array(X[i, :]))
            Y_list.append(np.array([1, 0], dtype=int))
            weights_list.append(weights[i] * 1.0/(n_neg + n_pos))
        for p in range(n_pos):
            X_list.append(X[i, :])
            Y_list.append(np.array([0, 1], dtype=int))
            weights_list.append(weights[i] * 1.0/(n_neg + n_pos))

    X_return = np.vstack(X_list)
    y_return = np.array(Y_list)
    weights_return = np.array(weights_list)
    return X_return, y_return, weights_return


if __name__ == '__main__':
    main()

