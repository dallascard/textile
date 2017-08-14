import os
import sys
import tempfile

import numpy as np
import tensorflow as tf

from ..models import tf_common
from ..models import evaluation
from ..util import file_handling as fh

class MLP:
    """
    Multilayer perceptron (representing documents as weighted sums of word vectors)
    """
    def __init__(self, dimensions, loss_function='log', nonlinearity='tanh', penalty=None, reg_strength=1e-3, output_dir=None, name='model'):
        self._model_type = 'MLP'
        self._dimensions = dimensions[:]
        self._loss_function = loss_function
        self._nonlinearity = nonlinearity
        self._reg_strength = reg_strength
        self._penalty = penalty
        self._n_classes = None
        if output_dir is None:
            self._output_dir = tempfile.gettempdir()
        else:
            self._output_dir = output_dir
        self._name = name
        self._train_f1 = None
        self._train_acc = None
        self._dev_f1 = None
        self._dev_acc = None

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None

    def get_model_type(self):
        return self._model_type

    def set_model(self, model, train_proportions, n_classes):
        self._train_proportions = train_proportions
        self._n_classes = n_classes
        if model is None:
            self._model = None
        else:
            self._model = model

    def fit(self, X_train, Y_train, X_dev, Y_dev, train_weights=None, dev_weights=None, seed=None, **kwargs):
        """
        Fit a classifier to data
        """
        _, n_classes = Y_train.shape
        self._n_classes = n_classes

        # store the proportion of class labels in the training data
        if train_weights is None:
            class_sums = np.sum(Y_train, axis=0)
        else:
            class_sums = np.dot(train_weights, Y_train) / train_weights.sum()
        self._train_proportions = (class_sums / float(class_sums.sum())).tolist()

        # if there is only a single type of label, make a default prediction
        train_labels = np.argmax(Y_train, axis=1)
        if np.max(self._train_proportions) == 1.0:
            self._model = None
        else:
            model_filename = os.path.join(self._output_dir, self._name + '.ckpt')
            self._model = tf_MLP(self._dimensions,  model_filename, loss_function=self._loss_function, penalty=self._penalty, reg_strength=self._reg_strength, nonlinearity=self._nonlinearity, seed=seed)
            self._model.train(X_train, Y_train, X_dev, Y_dev, train_weights, dev_weights)

        # do a quick evaluation and store the results internally
        train_pred = self.predict(X_train)
        self._train_acc = evaluation.acc_score(train_labels, train_pred, n_classes=n_classes, weights=train_weights)
        self._train_f1 = evaluation.f1_score(train_labels, train_pred, n_classes=n_classes, weights=train_weights)

        if X_dev is not None and Y_dev is not None:
            dev_labels = np.argmax(Y_dev, axis=1)
            dev_pred = self.predict(X_dev)
            self._dev_acc = evaluation.acc_score(dev_labels, dev_pred, n_classes=n_classes, weights=dev_weights)
            self._dev_f1 = evaluation.f1_score(dev_labels, dev_pred, n_classes=n_classes, weights=dev_weights)

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model is None:
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * np.argmax(self._train_proportions)
        else:
            return self._model.predict(X)

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
            model_probs = self._model.predict_probs(X)
            return model_probs

    def get_penalty(self):
        return self._penalty

    def get_reg_strength(self):
        return self._reg_strength

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
        return None

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

    def get_name(self):
        return self._name

    def get_dimensions(self):
        return self._dimensions[:]

    def save(self):
        output = {'model_type': self.get_model_type(),
                  'name': self.get_name(),
                  'dimensions': self.get_dimensions(),
                  'loss_function': self._loss_function,
                  'nonlinearity': self._nonlinearity,
                  'reg_strength': self.get_reg_strength(),
                  'penalty': self.get_penalty(),
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  'train_f1': self._train_f1,
                  'train_acc': self._train_acc,
                  'dev_f1': self._dev_f1,
                  'dev_acc': self._dev_acc
                  }
        fh.write_to_json(output, os.path.join(self._output_dir, self._name + '_metadata.json'), sort_keys=False)


def load_from_file(model_dir, name):
    input = fh.read_json(os.path.join(model_dir, name + '_metadata.json'))
    dimensions = input['dimensions']
    penalty = input['penalty']
    reg_strength = float(input['reg_strength'])
    n_classes = int(input['n_classes'])
    train_proportions = input['train_proportions']
    loss_function = input['loss_function']
    nonlinearity = input['nonlinearity']

    classifier = MLP(dimensions, loss_function, nonlinearity, penalty, reg_strength, model_dir, name=name)
    model_filename = os.path.join(model_dir, name + '.ckpt')
    model = tf_MLP(dimensions, model_filename, loss_function, penalty, reg_strength, nonlinearity)
    classifier.set_model(model, train_proportions, n_classes)
    return classifier


class tf_MLP:

    # TODO: optionally add embedding layer, or embedding updates, or attention over embeddings
    def __init__(self, dimensions, filename, loss_function='log', penalty=None, reg_strength=0.1, nonlinearity='tanh', seed=None):
        """
        Create an MLP in tensorflow, using a softmax on the final layer
        """
        self.dimensions = dimensions
        self.n_hidden_layers = len(dimensions) - 2
        self.loss_function = loss_function
        self.penalty = penalty
        self.reg_strength = reg_strength
        self.nonlinearity = nonlinearity
        self.filename = filename

        # create model
        self.x = tf.placeholder(tf.float32, shape=[None, dimensions[0]])
        self.y = tf.placeholder(tf.int32, shape=[None, dimensions[-1]])
        self.sample_weights = tf.placeholder(tf.float32)

        self.weights = []
        self.biases = []
        for layer in range(1, len(dimensions)):
            self.weights.append(tf_common.weight_variable((dimensions[layer-1], dimensions[layer]), name='weights' + str(layer), seed=seed))
            self.biases.append(tf_common.bias_variable((dimensions[layer], ), 0.0, name='weights' + str(layer)))

        self.scores = []
        self.scores.append(tf.matmul(self.x, self.weights[0]) + self.biases[0])
        for layer in range(1, len(self.weights)):
            if nonlinearity == 'tanh':
                self.scores.append(tf.matmul(tf.nn.tanh(self.scores[layer-1]), self.weights[layer]) + self.biases[layer])
            elif nonlinearity == 'sigmoid':
                self.scores.append(tf.matmul(tf.nn.sigmoid(self.scores[layer-1]), self.weights[layer]) + self.biases[layer])
            elif nonlinearity == 'relu':
                self.scores.append(tf.matmul(tf.nn.relu(self.scores[layer-1]), self.weights[layer]) + self.biases[layer])
            else:
                sys.exit("%s nonlinearity not recognized" % nonlinearity)

        self.scores_out = self.scores[-1]
        self.probs = tf.nn.softmax(logits=self.scores_out)

        if loss_function == 'log':
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.scores_out))
        elif loss_function == 'brier':
            self.loss = tf.reduce_mean(tf.square(self.y - self.probs))
        else:
            sys.exit("%s loss not supported" % loss_function)

        if penalty == 'l2':
            self.regularizer = tf.reduce_sum([tf.reduce_mean(tf.nn.l2_loss(w)) for w in self.weights])
            self.loss += reg_strength * self.regularizer
        elif penalty == 'l1':
            sys.exit('L1 regularization not supported')

        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(self.loss * self.sample_weights)
        self.initializer = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def train(self, X_train, Y_train, X_dev, Y_dev, w_train=None, w_dev=None, display_period=200, min_epochs=5, patience=5):
        done = False
        best_dev_f1 = 0

        n_train_items, n_classes = Y_train.shape
        n_dev_items, _ = Y_dev.shape
        _, n_features = X_train.shape
        if w_train is None:
            w_train = np.ones(n_train_items)
        if w_dev is None:
            w_dev = np.ones(n_dev_items)

        with tf.Session() as sess:
            sess.run(self.initializer)
            epoch = 1
            epochs_since_improvement = 0
            while not done:
                print("Starting epoch %d" % epoch)
                print("Iter\tLoss\tAccuracy")
                running_loss = 0.0
                running_accuracy = 0.0
                # just do minibatches sizes of 1 for now
                order = np.arange(n_train_items)
                np.random.shuffle(order)
                for i in order:
                    x_i = X_train[i, :].reshape((1, n_features))
                    y_i = np.array(Y_train[i], dtype=np.int32).reshape((1, n_classes))
                    w_i = w_train[i]

                    feed_dict = {self.x: x_i, self.y: y_i, self.sample_weights: w_i}
                    _, loss, scores = sess.run([self.train_step, self.loss, self.scores_out], feed_dict=feed_dict)
                    running_accuracy += (np.argmax(y_i, axis=1) == np.argmax(scores, axis=1)) * w_i
                    running_loss += loss * w_i
                    if i % display_period == 0 and i > 0:
                        print(i, running_loss / np.sum(w_train[:i+1]), running_accuracy / np.sum(w_train[:i+1]))

                predictions = []
                for i in range(n_dev_items):
                    x_i = X_dev[i, :].reshape((1, n_features))
                    y_i = np.array(Y_dev[i], dtype=np.int32).reshape((1, n_classes))
                    w_i = w_dev[i]

                    feed_dict = {self.x: x_i, self.sample_weights: w_i}
                    scores = sess.run(self.scores_out, feed_dict=feed_dict)
                    predictions.append(np.argmax(scores, axis=1))
                dev_acc = evaluation.acc_score(np.argmax(Y_dev, axis=1), predictions, n_classes=n_classes, weights=w_dev)
                dev_f1 = evaluation.f1_score(np.argmax(Y_dev, axis=1), predictions, n_classes=n_classes, weights=w_dev)
                print("Validation accuracy: %0.4f; f1: %0.4f" % (dev_acc, dev_f1))

                if dev_f1 > best_dev_f1:
                    print("New best validation f1; saving model")
                    best_dev_f1 = dev_f1
                    self.saver.save(sess, self.filename)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1
                    print("Epochs since improvement = %d" % epochs_since_improvement)

                if epoch >= min_epochs and epochs_since_improvement > patience:
                    print("Patience exceeded. Done")
                    print("Best validation f1 = %0.4f" % best_dev_f1)
                    done = True

                epoch += 1

    def predict(self, X):
        n_items, n_features = X.shape
        predictions = np.zeros(n_items, dtype=int)

        with tf.Session() as sess:
            self.saver.restore(sess, self.filename)
            # just do minibatche sizes of 1 for now
            for i in range(n_items):
                x_i = X[i, :].reshape((1, n_features))
                feed_dict = {self.x: x_i, self.sample_weights: 1.0}
                scores = sess.run(self.scores_out, feed_dict=feed_dict)
                pred = np.argmax(scores, axis=1)
                predictions[i] = pred

        return predictions

    def predict_probs(self, X):
        n_items, n_features = X.shape

        probs_list = []
        with tf.Session() as sess:
            self.saver.restore(sess, self.filename)
            # just do minibatche sizes of 1 for now
            for i in range(n_items):
                x_i = X[i, :].reshape((1, n_features))
                feed_dict = {self.x: x_i, self.sample_weights: 1.0}
                probs = sess.run(self.probs, feed_dict=feed_dict)
                probs_list.append(probs)

        pred_probs = np.array(probs_list, dtype=int)
        return pred_probs

    def get_n_params(self):
        n_params = 0
        for i in range(1, len(self.dimensions)):
            n_params += self.dimensions[i] * self.dimensions[i-1] + self.dimensions[i]
        return n_params
