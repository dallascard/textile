import os
import sys
import operator
import tempfile

import numpy as np
from scipy import sparse
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from ..util import file_handling as fh
from ..models import evaluation, calibration


class DecisionList:

    def __init__(self, depth=0, alpha=1.0, min_df=10, max_depth=50):
        self._depth = depth
        self._alpha = alpha
        self._min_df = min_df
        self._max_depth = max_depth
        self._feature_index = None
        self._feature_name = None
        self._present_obs = None
        self._absent_props = None
        self._child = None
        self._stoplist = []

    def fit(self, X, Y, w, all_col_names, reduced_col_names=None, interactive=True, stoplist=None):

        if reduced_col_names is None:
            reduced_col_names = all_col_names[:]
        if stoplist is None:
            stoplist = []

        command = 'y'
        while command != 'stop':
            n_remaining, n_features = X.shape

            # just record the overall observed proportions in the first node
            # get the label counts for each feature
            counts = np.zeros((2, n_features), dtype=float)
            for j in range(2):
                indices = Y[:, j] == 1
                counts[j, :] = X[indices, :].T.dot(w[indices]) + self._alpha

            # find the most predictive feature and record the observed label counts
            ratio = counts[1, :] / counts.sum(axis=0)
            most_predictive = np.argmax(ratio)
            name = reduced_col_names[most_predictive]

            if name in stoplist:
                print("Skipping %s" % name)
                col_index = np.ones(n_features, dtype=bool)
                col_index[most_predictive] = False
                X = X[:, col_index]
                reduced_col_names = [reduced_col_names[i] for i in range(len(col_index)) if i != most_predictive]

            else:
                print(name, counts[:, most_predictive])
                if interactive:
                    command = input("Inculde %s [y]/n/stop: " % name)
                else:
                    command = 'y'

                if command != 'n' and command != 'stop':
                    print("Including")
                    self._feature_name = reduced_col_names[most_predictive]
                    self._feature_index = all_col_names.index(self._feature_name)
                    self._present_obs = np.array([counts[0, most_predictive], counts[1, most_predictive]])

                    # subset the rows without that feature
                    if sparse.issparse(X):
                        absent_indices = np.array(X[:, most_predictive].todense()) == 0
                        absent_indices = np.reshape(absent_indices, (n_remaining, ))
                    else:
                        absent_indices = X[:, most_predictive] == 0

                    X = X[absent_indices, :]
                    Y = Y[absent_indices, :]
                    w = w[absent_indices]

                    # record the label proportion in the remaining items
                    absent_prop = np.sum(Y[:, 1] * w) / np.sum(w)
                    self._absent_props = np.array([2.0 - 2.0 * absent_prop, 2.0 * absent_prop])

                    # subset the features that meet the minimum frequency requirement
                    col_sums = X.sum(axis=0)
                    col_index = np.reshape(np.array(col_sums >= self._min_df), (n_features, ))
                    X = X[:, col_index]
                    reduced_col_names = [reduced_col_names[i] for i in range(len(col_index)) if col_index[i] > 0]

                    #print(self._feature_name, self._present_obs, self._absent_props)

                    if self._depth < self._max_depth:
                        list = DecisionList(self._depth + 1, alpha=self._alpha, min_df=self._min_df, max_depth=self._max_depth)
                        if list.fit(X, Y, w, all_col_names, reduced_col_names, interactive=interactive, stoplist=stoplist):
                            self._child = list
                    return True

                elif command != 'stop':
                    col_index = np.ones(n_features, dtype=bool)
                    col_index[most_predictive] = False
                    X = X[:, col_index]
                    reduced_col_names = [reduced_col_names[i] for i in range(len(col_index)) if i != most_predictive]
                    self._stoplist.append(name)

                elif command == 'stop':
                    return False

    def print_list(self):
        print(self._feature_name, ':', self._present_obs)
        if self._child is not None:
            self._child.print_list()
        else:
            print("Remainder:", self._absent_props[1] / np.sum(self._absent_props))

    def predict(self, X):
        n_items, n_features = X.shape
        predictions = np.zeros(n_items, dtype=int)
        for i in range(n_items):
            if sparse.issparse(X):
                prob_i = self._predict_probs_for_one_item(np.reshape(np.array(X[i, :].todense()), (n_features, )))
            else:
                prob_i = self._predict_probs_for_one_item(X[i, :])
            predictions[i] = int(prob_i > 0.5)
        return predictions

    def predict_probs(self, X):
        n_items, n_features = X.shape
        n_classes = 2
        probs = np.zeros((n_items, n_classes))
        for i in range(n_items):
            if sparse.issparse(X):
                prob_i = self._predict_probs_for_one_item(np.reshape(np.array(X[i, :].todense()), (n_features, )))
            else:
                prob_i = self._predict_probs_for_one_item(X[i, :])
            probs[i, :] = [1.0 - prob_i, prob_i]
        return probs

    def _predict_probs_for_one_item(self, x):
        obs_i = self._get_obs_for_one_item(x)
        return obs_i[1] / float(np.sum(obs_i))

    def _get_obs_for_one_item(self, x):
        #if self._feature_index < 0:
        #    return self._child._get_obs_for_one_item(x)
        if x[self._feature_index] > 0:
            return self._present_obs
        elif self._child is not None:
            return self._child._get_obs_for_one_item(x)
        else:
            return self._absent_props

    def get_betas(self, X):
        n_items, n_features = X.shape
        n_classes = 2
        betas = np.zeros((n_items, n_classes))
        for i in range(n_items):
            betas[i, :] = self._get_obs_for_one_item(X[i, :])
        return betas

    def collect_stopwords(self):
        stoplist = self._stoplist[:]
        if self._child is not None:
            stoplist.extend(self._child.collect_stopwords())
        return stoplist

    def test(self, X, Y, w, col_names, running_error=0.0, running_count=0.0):
        n_items, n_features = X.shape
        dev_prop = float(np.sum(Y[:, 1] * w) / np.sum(w))
        print("Observed proportions % 0.4f" % dev_prop)

        """
        if self._depth == 0:
            stopping_props = self._absent_props
            a = stopping_props[1]
            b = stopping_props[0]
            mean = (a / (a + b))
            var = a * b / ((a + b) ** 2 * (a + b + 1))
            stopping_error = np.abs(dev_prop - mean) * n_items
            stopping_var = var

            print("Stopping pred: %0.5f; Stopping error: %0.5f; Stopping var: %0.5f" % (mean, stopping_error, stopping_var))
            self._child.test(X, Y, w, col_names, 0.0, 0.0)
        """


        if sparse.issparse(X):
            present_indices = np.array(X[:, self._feature_index].todense()) > 0
            present_indices = np.reshape(present_indices, (n_items, ))
        else:
            present_indices = X[:, self._feature_index] > 0

        if np.sum(present_indices) > 0:
            dev_prop_present = np.sum(Y[present_indices, 1] * w[present_indices]) / np.sum(w[present_indices])
            pred_props = self._present_obs
            a = pred_props[1]
            b = pred_props[0]
            mean = a / (a + b)
            var = a * b / ((a + b) ** 2 * (a + b + 1))
            running_count += np.sum(present_indices)
            present_error = np.abs(dev_prop_present - mean) * np.sum(present_indices)
            present_var = var
            a = self._absent_props[1]
            b = self._absent_props[0]
            mean = (a / (a + b))
            var = a * b / ((a + b) ** 2 * (a + b + 1))
            default_error = np.abs(dev_prop_present - mean) * np.sum(present_indices)
            default_var = var
            print(self._feature_name, self._present_obs, self._present_obs[1] / np.sum(self._present_obs), Y[present_indices, :].sum(axis=0), dev_prop_present, present_error / float(np.sum(present_indices)), (running_error + present_error) / float(running_count), running_error + default_error)
        else:
            print("skipping %s" % self._feature_name)
            present_error = 0.0

        if sparse.issparse(X):
            absent_indices = np.array(X[:, self._feature_index].todense()) == 0
            absent_indices = np.reshape(absent_indices, (n_items, ))
        else:
            absent_indices = X[:, self._feature_index] == 0
        """
        dev_prop_absent = np.sum(Y[absent_indices, 1] * w[absent_indices]) / np.sum(w[absent_indices])
        absent_props = self._absent_props
        a = absent_props[1]
        b = absent_props[0]
        mean = a / (a + b)
        var = a * b / ((a + b) ** 2 * (a + b + 1))
        absent_se = np.abs(dev_prop_absent - mean) ** 2 * np.sum(absent_indices)
        absent_var = var * np.sum(absent_indices)
        """

        if self._child is not None:
            self._child.test(X[absent_indices, :], Y[absent_indices, :], w[absent_indices], col_names, running_error + present_error, running_count)


class DL:
    """
    Wrapper class for logistic regression from sklearn
    """
    def __init__(self, alpha=1.0, output_dir=None, name='model', pos_label=1, save_data=False):
        self._model_type = 'DL'
        self._alpha = alpha
        self._n_classes = None
        self._loss_function = None
        if output_dir is None:
            self._output_dir = tempfile.gettempdir()
        else:
            self._output_dir = output_dir
        self._name = name
        self._pos_label = pos_label
        self._train_f1 = None
        self._train_acc = None
        self._dev_f1 = None
        self._dev_acc = None
        self._save_data = save_data

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None
        # variable to hold the column names of the feature matrix
        self._col_names = None

    def get_model_type(self):
        return self._model_type

    def get_loss_function(self):
        return self._loss_function

    def set_model(self, model, train_proportions, col_names, n_classes):
        self._col_names = col_names
        self._train_proportions = train_proportions
        self._n_classes = n_classes
        if model is None:
            self._model = None
        else:
            self._model = model

    def fit(self, X_train, Y_train, train_weights=None, col_names=None, X_dev=None, Y_dev=None, dev_weights=None, min_df=5, max_depth=100, interactive=True, stoplist=None, *args, **kwargs):
        """
        Fit a classifier to data
        :param X: feature matrix: np.array(size=(n_items, n_features))
        :param Y: int matrix of item labels: (n_items, n_classes); each row is a one-hot vector
        :param train_weights: vector of item weights (one per item)
        :param col_names: names of the features (optional)
        :return: None
        """
        n_train_items, n_features = X_train.shape
        _, n_classes = Y_train.shape
        self._n_classes = n_classes
        assert n_classes == 2

        # store the proportion of class labels in the training data
        if train_weights is None:
            class_sums = np.sum(Y_train, axis=0)
            train_weights = np.ones(n_train_items)
        else:
            class_sums = np.dot(train_weights, Y_train) / train_weights.sum()
        self._train_proportions = (class_sums / float(class_sums.sum())).tolist()

        if col_names is not None:
            self._col_names = col_names
        else:
            self._col_names = range(n_features)

        # if there is only a single type of label, make a default prediction
        if np.max(self._train_proportions) == 1.0:
            self._model = None

        else:
            self._model = DecisionList(depth=0, alpha=self._alpha, min_df=min_df, max_depth=max_depth)

            self._model.fit(X_train, Y_train, train_weights, col_names, interactive=interactive, stoplist=stoplist)

            print("\nFinal model")
            self._model.print_list()

            if X_dev is not None and Y_dev is not None:
                dev_labels = np.argmax(Y_dev, axis=1)
                dev_pred = self.predict(X_dev)
                self._dev_acc = evaluation.acc_score(dev_labels, dev_pred, n_classes=n_classes, weights=dev_weights)
                self._dev_f1 = evaluation.f1_score(dev_labels, dev_pred, n_classes=n_classes, pos_label=self._pos_label, weights=dev_weights)
                self._dev_acc_cfm = calibration.compute_acc(dev_labels, dev_pred, n_classes, weights=dev_weights)
                self._dev_pvc_cfm = calibration.compute_pvc(dev_labels, dev_pred, n_classes, weights=dev_weights)

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        n_items, _ = X.shape
        if self._model is None:
            # else, get the model to make predictions
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
            return self._model.predict_probs(X)

    def predict_proportions(self, X=None, weights=None):
        pred_probs = self.predict_probs(X)
        predictions = np.argmax(pred_probs, axis=1)
        cc = calibration.cc(predictions, self._n_classes, weights)
        pcc = calibration.pcc(pred_probs, weights)
        acc = [0, 0]
        pvc = [0, 0]
        return cc, pcc, acc, pvc

    def test(self, X, Y, w):
        self._model.test(X, Y, w, self._col_names)

    def get_stoplist(self):
        if self._model is None:
            return []
        else:
            return self._model.collect_stopwords()

    def get_penalty(self):
        return None

    def get_alpha(self):
        return self._alpha

    def get_n_classes(self):
        return self._n_classes

    def get_train_proportions(self):
        return self._train_proportions

    def get_active_classes(self):
        return [0, 1]

    def get_default(self):
        return np.argmax(self._train_proportions)

    def get_col_names(self):
        return self._col_names

    def get_coefs(self, target_class=0):
        return None

    def get_intercept(self, target_class=0):
        return None

    def get_model_size(self):
        return 0

    def save(self):
        #print("Saving model")
        joblib.dump(self._model, os.path.join(self._output_dir, self._name + '.pkl'))
        all_coefs = {}
        all_intercepts = {}
        output = {'model_type': self._model_type,
                  'loss': self._loss_function,
                  'alpha': self.get_alpha(),
                  'penalty': self.get_penalty(),
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  'train_f1': self._train_f1,
                  'train_acc': self._train_acc,
                  'dev_f1': self._dev_f1,
                  'dev_acc': self._dev_acc,
                  'save_data': self._save_data
                  }
        fh.write_to_json(output, os.path.join(self._output_dir, self._name + '_metadata.json'), sort_keys=False)
        fh.write_to_json(self.get_col_names(), os.path.join(self._output_dir, self._name + '_col_names.json'), sort_keys=False)
        np.savez(os.path.join(self._output_dir, self._name + '_dev_info.npz'))


def load_from_file(model_dir, name):
    input = fh.read_json(os.path.join(model_dir, name + '_metadata.json'))
    col_names = fh.read_json(os.path.join(model_dir, name + '_col_names.json'))
    n_classes = int(input['n_classes'])
    alpha = float(input['alpha'])
    train_proportions = input['train_proportions']
    penalty = input['penalty']
    fit_intercept = input['fit_intercept']
    loss = input['loss']

    classifier = DL(alpha, output_dir=model_dir, name=name)
    model = joblib.load(os.path.join(model_dir, name + '.pkl'))
    classifier.set_model(model, train_proportions, col_names, n_classes)
    dev_info = np.load(os.path.join(model_dir, name + '_dev_info.npz'))
    classifier._dev_acc_cfm = dev_info['acc_cfm']
    classifier._dev_pvc_cfm = dev_info['pvc_cfm']
    classifier._venn_info = dev_info['venn_info']
    return classifier
