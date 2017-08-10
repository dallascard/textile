import os
import operator

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression as lr
import tensorflow as tf

from ..util import file_handling as fh
from ..models import tf_common

class MLP:
    """
    Multilayer perceptron (representing documents as weighted sums of word vectors)
    """
    def __init__(self, alpha=1e-3, penalty=None, fit_intercept=True, n_classes=2, n_layers=0):
        self._model_type = 'MLP'
        self._alpha = alpha
        self._penalty = penalty
        self._fit_intercept = fit_intercept
        self._n_classes = n_classes
        self._n_layers = n_layers

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None
        # variable to hold the column names of the feature matrix
        self._col_names = None

    def get_model_type(self):
        return self._model_type

    # TODO: revise
    def set_model(self, model, train_proportions, col_names):
        self._col_names = col_names
        self._train_proportions = train_proportions
        if model is None:
            self._model = None
        else:
            self._model = model

    # TODO: convert to MLP
    def fit(self, X_train, Y_train, X_dev, Y_dev, train_weights=None, dev_weights=None, col_names=None):
        """
        Fit a classifier to data
        :param X: feature matrix: np.array(size=(n_items, n_features))
        :param Y: one-hot label encoding np.array(size=(n_items, n_classes))
        :return: None
        """
        n_train_items, n_features = X_train.shape
        assert n_features == len(col_names)
        _, n_classes = Y_train.shape
        assert n_classes == self._n_classes

        # store the proportion of class labels in the training data
        class_sums = np.array(np.sum(Y_train, axis=0), dtype=float)
        self._train_proportions = (class_sums / np.sum(class_sums)).tolist()
        if col_names is not None:
            self._col_names = col_names
        else:
            self._col_names = range(n_features)

        # if there is only a single type of label, make a default prediction
        if np.max(self._train_proportions) == 1.0:
            self._model = None
        else:
            self._model = tf_MLP(0, [n_features, n_classes], fit_intercept=self._fit_intercept)
            #self._model = lr(penalty=self._penalty, C=self._alpha, fit_intercept=self._fit_intercept)
            # otherwise, train the model
            self._model.train(X_train, Y_train, X_dev, Y_dev, train_weights, dev_weights)

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model is None:
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * np.argmax(self._train_proportions)
        else:
            return self._model.predict(X)

    """
    def predict_probs(self, X):
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model is None:
            default = np.argmax(self._train_proportions)
            full_probs[:, default] = 1.0
            return full_probs
        else:
            # otherwise, get probabilities from the model
            model_probs = self._model.predict_proba(X)
            # map these probabilities back to the full set of classes
            for i, cl in enumerate(self._model.classes_):
                full_probs[:, cl] = model_probs[:, i]
            return full_probs
    """

    def get_penalty(self):
        return self._penalty

    def get_alpha(self):
        return self._alpha

    def get_n_classes(self):
        return self._n_classes

    def get_train_proportions(self):
        return self._train_proportions

    def get_active_classes(self):
        if self._model is None:
            return []
        else:
            return range(self._n_classes)

    def get_default(self):
        return np.argmax(self._train_proportions)

    def get_col_names(self):
        return self._col_names

    """
    def get_coefs(self, target_class=0):
        coefs = zip(self._col_names, np.zeros(len(self._col_names)))
        if self._model is not None:
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    coefs = zip(self._col_names, self._model.coef_[i])
                    break
        return coefs
    """

    """
    def get_intercept(self, target_class=0):
        # if we've saved a default value, there are no intercepts
        intercept = 0
        if self._model is not None:
            # otherwise, see if the model an intercept for this class
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    intercept = self._model.intercept_[i]
                    break
        return intercept
    """

    def get_model_size(self):
        if self._model is None:
            return 0
        else:
            return self._model.get_n_params()

    """
    def save(self, output_dir):
        #print("Saving model")
        joblib.dump(self._model, os.path.join(output_dir, 'model.pkl'))
        all_coefs = {}
        all_intercepts = {}
        # deal with the inconsistencies in sklearn depending on the number of classes
        if self._model is not None:
            if len(self.get_active_classes()) == 2:
                coefs_list = self.get_coefs(0)
                coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
                coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
                all_coefs[0] = coefs_sorted
                all_intercepts[0] = self.get_intercept(0)
            else:
                for cln, cl in enumerate(self.get_active_classes()):
                    coefs_list = self.get_coefs(cln)
                    coefs_dict = {k: v for (k, v) in coefs_list if v != 0}
                    coefs_sorted = sorted(coefs_dict.items(), key=operator.itemgetter(1))
                    all_coefs[str(cl)] = coefs_sorted
                    all_intercepts[str(cl)] = self.get_intercept(cl)
        output = {'model_type': 'LR',
                  'alpha': self.get_alpha(),
                  'penalty': self.get_penalty(),
                  'intercepts': all_intercepts,
                  'coefs': all_coefs,
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  'fit_intercept': self._fit_intercept
                  }
        fh.write_to_json(output, os.path.join(output_dir, 'metadata.json'), sort_keys=False)
        fh.write_to_json(self.get_col_names(), os.path.join(output_dir, 'col_names.json'), sort_keys=False)
    """


class tf_MLP:

    def __init__(self, n_hidden_layers, dimensions, penalty=None, reg_strength=0, nonlinearity='sigmoid', fit_intercept=True):
        """
        Create an MLP in tensorflow, using a softmax on the final layer
        """
        assert len(dimensions) == int(n_hidden_layers) + 2
        self.n_hidden_layers = n_hidden_layers
        self.dimensions = dimensions
        self.fit_intercept = fit_intercept

        # create model
        self.x = tf.placeholder(tf.float32, shape=[None, dimensions[0]])
        self.y = tf.placeholder(tf.int32, shape=[None, dimensions[-1]])
        self.sample_weights = tf.placeholder(tf.float32)

        self.w = tf_common.weight_variable((dimensions[0], dimensions[1]))
        self.b = tf_common.bias_variable((dimensions[1], ))
        if fit_intercept:
            self.s = tf.matmul(self.x, self.w) + self.b
        else:
            self.s = tf.matmul(self.x, self.w)

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.s))
        self.initializer = tf.global_variables_initializer()
        self.train_step = None
        self.saver = tf.train.Saver()

    def train(self, X_train, Y_train, X_dev, Y_dev, w_train=None, w_dev=None, step_size=0.1, display_period=200, n_epochs=10, min_epochs=5, max_epochs=50):
        # TODO: add regularization
        # TODO: shufling
        # TODO: save best model
        self.train_step = tf.train.GradientDescentOptimizer(step_size).minimize(self.cross_entropy * self.sample_weights)

        n_train_items, n_classes = Y_train.shape
        n_dev_items, _ = Y_dev.shape
        _, n_features = X_train.shape
        if w_train is None:
            w_train = np.ones(n_train_items)
        if w_dev is None:
            w_dev = np.ones(n_dev_items)

        with tf.Session() as sess:
            sess.run(self.initializer)
            for epoch in range(n_epochs):
                print("Starting epoch %d" % epoch)
                print("Iter\tLoss\tAccuracy")
                running_loss = 0.0
                running_accuracy = 0.0
                # just do minibatche sizes of 1 for now
                for i in range(n_train_items):
                    x_i = X_train[i, :].reshape((1, n_features))
                    y_i = np.array(Y_train[i], dtype=np.int32).reshape((1, n_classes))
                    w_i = w_train[i]

                    feed_dict = {self.x: x_i, self.y: y_i, self.sample_weights: w_i}
                    _, loss, scores = sess.run([self.train_step, self.cross_entropy, self.s], feed_dict=feed_dict)
                    running_accuracy += (np.argmax(y_i, axis=1) == np.argmax(scores, axis=1)) * w_i
                    running_loss += loss * w_i
                    if i % display_period == 0 and i > 0:
                        print(i, running_loss / np.sum(w_train[:i+1]), running_accuracy / np.sum(w_train[:i+1]))

                running_accuracy = 0.0
                for i in range(n_dev_items):
                    x_i = X_dev[i, :].reshape((1, n_features))
                    y_i = np.array(Y_dev[i], dtype=np.int32).reshape((1, n_classes))
                    w_i = w_dev[i]

                    feed_dict = {self.x: x_i, self.sample_weights: w_i}
                    scores = sess.run(self.s, feed_dict=feed_dict)
                    #print(y_i, scores)
                    running_accuracy += (np.argmax(y_i, axis=1) == np.argmax(scores, axis=1)) * w_i

                print("Dev accuracy:", running_accuracy / np.sum(w_dev))

            biases = sess.run(self.b)
            print("Biases:", biases)
            save_path = self.saver.save(sess, "/tmp/model.ckpt")
            print("Model saved in file: %s" % save_path)

    def predict(self, X):
        n_items, n_features = X.shape

        #predictions = np.zeros([n_items, n_classes], dtype=int)
        predictions = np.zeros(n_items, dtype=int)

        with tf.Session() as sess:
            self.saver.restore(sess, "/tmp/model.ckpt")
            # just do minibatche sizes of 1 for now
            for i in range(n_items):
                x_i = X[i, :].reshape((1, n_features))
                feed_dict = {self.x: x_i, self.sample_weights: 1.0}
                scores = sess.run(self.s, feed_dict=feed_dict)
                pred = np.argmax(scores, axis=1)
                #predictions[i, pred] = 1
                predictions[i] = pred

        return predictions

    def get_n_params(self):
        n_params = 0
        for i in range(1, len(self.dimensions)):
            n_params += self.dimensions[i] * self.dimensions[i-1]
            if self.fit_intercept:
                n_params += self.dimensions[i]
        return n_params
