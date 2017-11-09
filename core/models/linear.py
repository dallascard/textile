import os
import sys
import operator
import tempfile

import numpy as np
from scipy.special import expit
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from ..util import file_handling as fh
from ..models import evaluation, calibration
from ..main import train
from .core.models import lrb


class LinearClassifier:
    """
    Wrapper class for logistic regression from sklearn
    """
    def __init__(self, alpha, penalty='l2', fit_intercept=True, output_dir=None, name='model', pos_label=1, save_data=False, do_cfm=False, do_platt=False, lower=None):
        self._model_type = 'LR'
        self._alpha = alpha
        self._penalty = penalty
        self._fit_intercept = fit_intercept
        self._n_classes = None
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
        self._dev_acc_cfm = None
        self._dev_acc_cfms_ms = None
        self._venn_info = None
        self._save_data = save_data
        self._X_train = None
        self._Y_train = None
        self._w_train = None
        self._do_cfm = do_cfm
        self._do_platt = do_platt
        self._platt_a = None
        self._platt_b = None
        self._platt_T = None
        self._lower = lower

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None
        # variable to hold the column names of the feature matrix
        self._col_names = None

    def get_model_type(self):
        return self._model_type

    def set_model(self, model, train_proportions, col_names, n_classes, X_train=None, Y_train=None, train_weights=None):
        self._col_names = col_names
        self._train_proportions = train_proportions
        self._n_classes = n_classes
        self._X_train = X_train
        self._Y_train = Y_train
        self._w_train = train_weights
        if model is None:
            self._model = None
        else:
            self._model = model

    def fit(self, X_train, Y_train, train_weights=None, col_names=None):
        """
        Fit a classifier to data
        :param X: feature matrix: np.array(size=(n_items, n_features))
        :param Y: int matrix of item labels: (n_items, n_classes); each row is a 1-hot vector
        :param train_weights: vector of item weights (one per item)
        :param col_names: names of the features (optional)
        :return: None
        """
        X_train, Y_train, train_weights = train.prepare_data(X_train, Y_train, train_weights, loss='log')

        n_train_items, n_features = X_train.shape
        _, n_classes = Y_train.shape
        self._n_classes = n_classes

        # store the proportion of class labels in the training data
        if train_weights is None:
            class_sums = np.sum(Y_train, axis=0)
        else:
            class_sums = np.dot(train_weights, Y_train) / train_weights.sum()
        self._train_proportions = (class_sums / float(class_sums.sum())).tolist()

        if col_names is not None:
            self._col_names = col_names
        else:
            self._col_names = range(n_features)

        # if there is only a single type of label, make a default prediction
        # NOTE: this assumes that all items are singly labeled!
        train_labels = np.argmax(Y_train, axis=1).reshape((n_train_items, ))
        if np.max(self._train_proportions) == 1.0:
            self._model = None

        else:
            if self._lower is None:
                self._model = LogisticRegression(penalty=self._penalty, C=self._alpha, fit_intercept=self._fit_intercept)
            else:
                assert self._penalty == 'l1'
                self._model = lrb.LogisticRegressionBounded(C=self._alpha, fit_intercept=self._fit_intercept, lower=self._lower)

            # train the model using a vector of labels
            self._model.fit(X_train, train_labels, sample_weight=train_weights)

        # do a quick evaluation and store the results internally
        train_pred = self.predict(X_train)
        self._train_acc = evaluation.acc_score(train_labels, train_pred, n_classes=n_classes, weights=train_weights)
        self._train_f1 = evaluation.f1_score(train_labels, train_pred, n_classes=n_classes, pos_label=self._pos_label, weights=train_weights)

        if self._save_data:
            self._X_train = X_train.copy()
            self._Y_train = Y_train.copy()
            self._w_train = train_weights.copy()

    def fit_cfms(self, X_dev, Y_dev, dev_weights, prep_data=False):
        _, n_classes = Y_dev.shape
        if prep_data:
            X_dev, Y_dev, dev_weights = train.prepare_data(X_dev, Y_dev, dev_weights, loss='log')

        dev_labels = np.argmax(Y_dev, axis=1)
        dev_pred = self.predict(X_dev)
        dev_pred_probs = self.predict_probs(X_dev)
        self._dev_acc = evaluation.acc_score(dev_labels, dev_pred, n_classes=n_classes, weights=dev_weights)
        self._dev_f1 = evaluation.f1_score(dev_labels, dev_pred, n_classes=n_classes, pos_label=self._pos_label, weights=dev_weights)
        self._dev_acc_cfm = calibration.compute_acc(dev_labels, dev_pred, n_classes, weights=dev_weights)
        self._dev_acc_cfms_ms = calibration.compute_acc_median_sweep(dev_labels, dev_pred_probs, n_classes, weights=dev_weights)

    def fit_platt(self, X_dev, Y_dev, dev_weights, prep_data=False):
        if prep_data:
            X_dev, Y_dev, dev_weights = train.prepare_data(X_dev, Y_dev, dev_weights, loss='log')
        n_dev, n_classes = Y_dev.shape

        if n_classes == 2:
            # fit a platt model with an intercept
            scores = np.reshape(self.score(X_dev), (n_dev, 1))

            model = train.train_lr_model_with_cv(scores, Y_dev, dev_weights, col_names=['p'], basename='platt2', intercept=True, n_dev_folds=2, pos_label=self._pos_label, do_ensemble=False, fit_platt=False, fit_cfms=False, save_model=False, verbose=False)
            coefs = dict(model.get_coefs())
            self._platt_a = coefs['p']
            self._platt_b = model.get_intercept()

            # fit a platt model without an intercept
            model = train.train_lr_model_with_cv(scores, Y_dev, dev_weights, col_names=['p'], basename='platt1', intercept=False, n_dev_folds=2, pos_label=self._pos_label, do_ensemble=False, fit_platt=False, fit_cfms=False, save_model=False, verbose=False)
            self._platt_T = dict(model.get_coefs())['p']

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model is None:
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * np.argmax(self._train_proportions)
        else:
            return self._model.predict(X)

    def predict_probs(self, X, do_platt=False):
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model is None:
            default = np.argmax(self._train_proportions)
            full_probs[:, default] = 1.0
            return full_probs
        else:
            if do_platt and self._n_classes == 2:
                scores = self.score(X)
                corrected_scores = expit(self._platt_a * scores + self._platt_b)
                full_probs[:, 0] = 1 - corrected_scores
                full_probs[:, 1] = corrected_scores
            else:
                # otherwise, get probabilities from the model
                model_probs = self._model.predict_proba(X)
                # map these probabilities back to the full set of classes
                for i, cl in enumerate(self._model.classes_):
                    full_probs[:, cl] = model_probs[:, i]
            return full_probs


    def score(self, X):
        return self._model.decision_function(X)

    def predict_proportions(self, X=None, weights=None, do_cfm=False, do_platt=False):
        pred_probs = self.predict_probs(X)
        predictions = np.argmax(pred_probs, axis=1)
        if do_cfm:
            assert self._do_cfm
            if self._n_classes == 2:
                acc = calibration.apply_acc_binary(predictions, self._dev_acc_cfm, weights)
                acc_ms = calibration.apply_acc_binary_median_sweep(pred_probs, self._dev_acc_cfms_ms, weights)
            else:
                acc = calibration.apply_acc_bounded_lstsq(predictions, self._dev_acc_cfm)
                acc_ms = [0, 0]
            return acc, acc_ms
        elif do_platt:
            assert self._do_platt
            scores = self.score(X)
            corrected_scores = expit(self._platt_T * scores)
            corrected_probs = np.zeros_like(pred_probs)
            corrected_probs[:, 0] = 1 - corrected_scores
            corrected_probs[:, 1] = corrected_scores
            platt1 = calibration.pcc(corrected_probs, weights)

            corrected_scores = expit(self._platt_a * scores + self._platt_b)
            corrected_probs[:, 0] = 1 - corrected_scores
            corrected_probs[:, 1] = corrected_scores
            platt2 = calibration.pcc(corrected_probs, weights)
            return platt1, platt2
        else:
            cc = calibration.cc(predictions, self._n_classes, weights)
            pcc = calibration.pcc(pred_probs, weights)
            return cc, pcc

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
            return self._model.classes_

    def get_default(self):
        return np.argmax(self._train_proportions)

    def get_col_names(self):
        return self._col_names

    def get_coefs(self, target_class=0):
        coefs = zip(self._col_names, np.zeros(len(self._col_names)))
        if self._model is not None:
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    coefs = zip(self._col_names, self._model.coef_[i])
                    break
        return coefs

    def get_intercept(self, target_class=0):
        # if we've saved a default value, there are no intercepts
        intercept = 0
        if self._model is not None:
            # otherwise, see if the model has an intercept for this class
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    intercept = self._model.intercept_[i]
                    break
        return intercept

    def get_model_size(self):
        n_nonzeros_coefs = 0
        if self._model is None:
            return 0
        else:
            coefs = self._model.coef_
            for coef_list in coefs:
                n_nonzeros_coefs += np.sum([1.0 for c in coef_list if c != 0])
            return n_nonzeros_coefs

    def save(self):
        #print("Saving model")
        joblib.dump(self._model, os.path.join(self._output_dir, self._name + '.pkl'))
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
                  'fit_intercept': self._fit_intercept,
                  'pos_label': self._pos_label,
                  'train_f1': self._train_f1,
                  'train_acc': self._train_acc,
                  'dev_f1': self._dev_f1,
                  'dev_acc': self._dev_acc,
                  'save_data': self._save_data
                  }
        fh.write_to_json(output, os.path.join(self._output_dir, self._name + '_metadata.json'), sort_keys=False)
        fh.write_to_json(self.get_col_names(), os.path.join(self._output_dir, self._name + '_col_names.json'), sort_keys=False)
        np.savez(os.path.join(self._output_dir, self._name + '_dev_info.npz'), acc_cfm=self._dev_acc_cfm, acc_cfms_ms= self._dev_acc_cfms_ms)
        if self._save_data:
            np.savez(os.path.join(self._output_dir, self._name + '_training_data.npz'), X_train=self._X_train, Y_train=self._Y_train, train_weights=self._w_train)


def load_from_file(model_dir, name):
    input = fh.read_json(os.path.join(model_dir, name + '_metadata.json'))
    col_names = fh.read_json(os.path.join(model_dir, name + '_col_names.json'))
    n_classes = int(input['n_classes'])
    alpha = float(input['alpha'])
    train_proportions = input['train_proportions']
    penalty = input['penalty']
    fit_intercept = input['fit_intercept']
    pos_label = input['pos_label']

    save_data = False
    if 'save_data' in input:
        save_data = input['save_data']

    X_train = None
    Y_train = None
    train_weights = None
    if save_data:
        training_data = np.load(os.path.join(model_dir, name + '_training_data.npz'))
        X_train = training_data['X_train']
        Y_train = training_data['Y_train']
        train_weights = training_data['training_weights']

    classifier = LinearClassifier(alpha, penalty, fit_intercept, output_dir=model_dir, name=name, pos_label=pos_label, save_data=save_data)
    model = joblib.load(os.path.join(model_dir, name + '.pkl'))
    classifier.set_model(model, train_proportions, col_names, n_classes, X_train, Y_train, train_weights)
    dev_info = np.load(os.path.join(model_dir, name + '_dev_info.npz'))
    classifier._dev_acc_cfm = dev_info['acc_cfm']
    classifier._dev_acc_cfms_ms = dev_info['acc_cfms_ms']
    #classifier._venn_info = dev_info['venn_info']
    return classifier
