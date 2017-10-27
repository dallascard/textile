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
from ..main import train


class DecisionList:

    def __init__(self, alpha=1.0, max_depth=50):
        self._alpha = alpha
        self._max_depth = max_depth
        self._feature_indices = None
        self._feature_names = None
        self._present_obs = None
        self._absent_props = None
        self._stoplist = []
        self._resid_model = None
        self._col_names = None

    def fit(self, X, Y, w, feature_list, all_col_names):
        self._feature_names = []
        self._feature_indices = []
        self._present_obs = np.zeros([self._max_depth, 2])
        self._absent_props = np.zeros([self._max_depth, 2])

        self._col_names = all_col_names
        col_names = all_col_names[:]

        depth = 0
        list_index = 0
        while depth < self._max_depth:
            n_items, n_features = X.shape
            feature = feature_list[list_index]

            if feature not in col_names:
                print("Skipping %s" % feature)
                list_index += 1

            if feature in col_names:
                self._feature_names.append(feature)
                feature_index = col_names.index(feature)
                orig_index = all_col_names.index(feature)
                self._feature_indices.append(orig_index)

                # get the label counts for each feature
                counts = np.zeros((2, n_features), dtype=float)
                for j in range(2):
                    indices = Y[:, j] == 1
                    counts[j, :] = X[indices, :].T.dot(w[indices]) + self._alpha

                self._present_obs[depth, :] = np.array([counts[0, feature_index], counts[1, feature_index]])

                # subset the rows without that feature
                if sparse.issparse(X):
                    absent_indices = np.array(X[:, feature_index].todense()) == 0
                    absent_indices = np.reshape(absent_indices, (n_items, ))
                else:
                    absent_indices = X[:, feature_index] == 0

                X = X[absent_indices, :]
                Y = Y[absent_indices, :]
                w = w[absent_indices]

                # record the label proportion in the remaining items
                absent_prop = np.sum(Y[:, 1] * w) / np.sum(w)
                self._absent_props[depth, :] = np.array([2.0 - 2.0 * absent_prop, 2.0 * absent_prop])

                print("Adding", feature, self._present_obs[depth, :], absent_prop)

                # subset the features that meet the minimum frequency requirement
                col_index = np.ones(n_features, dtype=bool)
                col_index[feature_index] = False
                X = X[:, col_index]
                col_names = [col_names[i] for i in range(len(col_index)) if col_index[i] > 0]

                depth += 1
                list_index += 1


    """
    def fit_interactive(self, X, Y, w, all_col_names, reduced_col_names=None, stoplist=None):

        if reduced_col_names is None:
            reduced_col_names = all_col_names[:]
        if stoplist is None:
            stoplist = []

        command = 'y'
        while command != 'stop':
            n_remaining, n_features = X.shape

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
                command = input("Inculde %s [y]/n/stop: " % name)

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
                        if list.fit_interactive(X, Y, w, all_col_names, reduced_col_names, stoplist=stoplist):
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
    """

    def print_list(self):
        for depth in range(self._max_depth):
            print(self._feature_names[depth], ':', self._present_obs[depth, :], self._absent_props[depth, :])

    def predict(self, X):
        probs = np.array(self.predict_probs(X))
        predictions = np.argmax(probs, axis=1)
        return predictions

    def predict_probs(self, X):
        n_items, n_features = X.shape

        X_copy = X.copy()
        for feature in self._feature_names:
            feature_index = self._col_names.index(feature)
            X_copy[:, feature_index] = 0

        # apply the residual prediction to all items
        probs = self._resid_model.predict_probs(X_copy)
        print(np.mean(probs, axis=0))

        # then go through the decision list in reverse order, assigning observed proportions to those items
        for depth in range(self._max_depth-1, -1, -1):
            feature = self._feature_names[depth]
            feature_index = self._col_names.index(feature)
            if sparse.issparse(X):
                selector = np.array(np.array(X[:, feature_index].todense()) > 0, dtype=bool).reshape((n_items, ))
            else:
                selector = np.array(X[:, feature_index] > 0, dtype=bool)
            probs[selector, :] = self._present_obs[depth, :] / np.sum(self._present_obs[depth, :])
        print(np.mean(probs, axis=0))
        return probs

    """
    def _predict_probs_for_one_item(self, x):
        obs_i = self._get_obs_for_one_item(x)
        return obs_i[1] / float(np.sum(obs_i))

    def _get_obs_for_one_item(self, x):
        n_features = len(x)
        for depth in range(self._max_depth):
            if x[self._feature_indices[depth]] > 0:
                return self._present_obs[depth, :]
        pred_probs = self._resid_model.predict_probs(x.reshape(1, n_features))
        return [1.0 - pred_probs, pred_probs]
    """

    """
    def get_betas(self, X):
        n_items, n_features = X.shape
        n_classes = 2
        betas = np.zeros((n_items, n_classes))
        for i in range(n_items):
            betas[i, :] = self._get_obs_for_one_item(X[i, :])
        return betas
    """

    def train_resid(self, X_all, Y_all, w_all, col_names, name, output_dir=None, n_classes=2, objective='f1', penalty='l1', pos_label=1, do_ensemble=True, save_model=True):
        self._resid_model = train.train_lr_model_with_cv(X_all, Y_all, w_all, col_names, name, output_dir=output_dir, n_classes=n_classes, objective=objective, loss='log', penalty=penalty, intercept=True, n_dev_folds=5, alpha_min=0.01, alpha_max=1000.0, n_alphas=8, pos_label=pos_label, do_ensemble=do_ensemble, prep_data=False, save_model=save_model)

    def test(self, X, Y, w, col_names, running_error=0.0, running_count=0.0):

        dev_prop = float(np.sum(Y[:, 1] * w) / np.sum(w))
        print("Observed proportions % 0.4f" % dev_prop)

        for depth in range(self._max_depth):
            n_items, n_features = X.shape
            if sparse.issparse(X):
                present_indices = np.array(X[:, self._feature_indices[depth]].todense()) > 0
                present_indices = np.reshape(present_indices, (n_items, ))
            else:
                present_indices = X[:, self._feature_indices[depth]] > 0

            if np.sum(present_indices) > 0:
                dev_prop_present = np.sum(Y[present_indices, 1] * w[present_indices]) / np.sum(w[present_indices])
                pred_props = self._present_obs[depth, :]
                a = pred_props[1]
                b = pred_props[0]
                mean = a / (a + b)
                var = a * b / ((a + b) ** 2 * (a + b + 1))
                running_count += np.sum(present_indices)
                present_error = np.abs(dev_prop_present - mean) * np.sum(present_indices)
                present_var = var
                a = self._absent_props[depth, 1]
                b = self._absent_props[depth, 0]
                mean = (a / (a + b))
                var = a * b / ((a + b) ** 2 * (a + b + 1))
                default_error = np.abs(dev_prop_present - mean) * np.sum(present_indices)
                default_var = var
                print(self._feature_names[depth], self._present_obs[depth, :], self._present_obs[depth, 1] / np.sum(self._present_obs[depth, :]), Y[present_indices, :].sum(axis=0), dev_prop_present, present_error / float(np.sum(present_indices)), (running_error + present_error) / float(running_count), running_error + default_error)

            if sparse.issparse(X):
                absent_indices = np.array(X[:, self._feature_indices[depth]].todense()) == 0
                absent_indices = np.reshape(absent_indices, (n_items, ))
            else:
                absent_indices = X[:, self._feature_indices[depth]] == 0

            X = X[absent_indices, :]
            Y = Y[absent_indices, :]
            w = w[absent_indices]



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
        # variable to hold the residual LR model
        self._resid_model = None
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

    def feature_selection(self, X, Y, w, orig_col_names, max_features=50, interactive=False, stoplist=None):

        _, n_classes = Y.shape
        self._n_classes = n_classes
        assert n_classes == 2

        col_names = orig_col_names[:]

        features = []

        while len(features) < max_features:
            n_items, n_features = X.shape
            # get the label counts for each feature
            counts = np.zeros((2, n_features), dtype=float)
            for j in range(2):
                indices = Y[:, j] == 1
                counts[j, :] = X[indices, :].T.dot(w[indices]) + self._alpha

            ratio = counts[1, :] / counts.sum(axis=0)
            index = int(np.argmax(ratio))
            feature = col_names[index]

            if feature in stoplist:
                print("Skipping", feature)
            else:
                print("Selecting", feature, counts[:, index])
                features.append(feature)

                # subset the rows without that feature
                if sparse.issparse(X):
                    absent_indices = np.array(X[:, index].todense()) == 0
                    absent_indices = np.reshape(absent_indices, (n_items, ))
                else:
                    absent_indices = X[:, index] == 0

                X = X[absent_indices, :]
                Y = Y[absent_indices, :]
                w = w[absent_indices]

            # remove the feature
            col_index = np.ones(n_features, dtype=bool)
            col_index[index] = False
            X = X[:, col_index]
            col_names = [col_names[i] for i in range(len(col_index)) if col_index[i] > 0]

        return features

    def fit(self, X_train, Y_train, train_weights=None, col_names=None, feature_list=None, X_dev=None, Y_dev=None, dev_weights=None, max_features=100, interactive=False, stoplist=None, objective='f1', penalty='l1'):
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

        print("X_train.shape", X_train.shape)
        print("X_dev.shape", X_dev.shape)

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
            if feature_list is None:
                if X_dev is not None and Y_dev is not None:
                    print("Using dev data to do feature selection")
                    feature_list = self.feature_selection(X_dev, Y_dev, dev_weights, col_names, max_features, interactive, stoplist)
                else:
                    print("Using training data to do feature selection (double-dipping?)")
                    feature_list = self.feature_selection(X_train, Y_train, train_weights, col_names, max_features, interactive, stoplist)

            self._model = DecisionList(alpha=self._alpha, max_depth=max_features)
            self._model.fit(X_train, Y_train, train_weights, feature_list, col_names)

            print("X_train.shape", X_train.shape)
            print("X_dev.shape", X_dev.shape)

            # Now fit a basic LR model to the remaining features
            if X_dev is not None:
                if sparse.issparse(X_train):
                    X_all = sparse.vstack([X_train, X_dev])
                else:
                    X_all = np.vstack([X_train, X_dev])
                Y_all = np.vstack([Y_train, Y_dev])
                w_all = np.r_[train_weights, dev_weights]
            else:
                X_all = X_train
                Y_all = Y_train
                w_all = train_weights

            print("X_all.shape", X_all.shape)

            # remove items that have features in our feature list
            for feature in self._model._feature_names:
                n_items, n_features = X_all.shape
                feature_index = self._model._col_names.index(feature)
                if sparse.issparse(X_all):
                    selector = np.array(np.array(X_all[:, feature_index].todense()) == 0, dtype=bool).reshape((n_items, ))
                else:
                    selector = np.array(X_all[:, feature_index] == 0, dtype=bool)
                X_all = X_all[selector, :]
                Y_all = Y_all[selector, :]
                w_all = w_all[selector]

            print("X_all.shape", X_all.shape)

            self._model.train_resid(X_all, Y_all, w_all, col_names, self._name + '_resid', output_dir=self._output_dir, n_classes=2, objective='calibration', penalty='l1', pos_label=1, do_ensemble=True, save_model=True)

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
