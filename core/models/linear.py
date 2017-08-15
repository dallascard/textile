import os
import sys
import operator
import tempfile

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression, Lasso, Ridge

from ..util import file_handling as fh
from ..models import evaluation

class LinearClassifier:
    """
    Wrapper class for logistic regression from sklearn
    """
    def __init__(self, alpha, loss_function='log', penalty='l2', fit_intercept=True, output_dir=None, name='model'):
        self._model_type = 'LR'
        self._alpha = alpha
        self._loss_function = loss_function
        self._penalty = penalty
        self._fit_intercept = fit_intercept
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

    def fit(self, X_train, Y_train, train_weights=None, col_names=None, X_dev=None, Y_dev=None, dev_weights=None, *args, **kwargs):
        """
        Fit a classifier to data
        :param X: feature matrix: np.array(size=(n_items, n_features))
        :param Y: int matrix of item labels: (n_items, n_classes); each row is a 1-hot vector
        :param train_weights: vector of item weights (one per item)
        :param col_names: names of the features (optional)
        :return: None
        """
        n_train_items, n_features = X_train.shape
        _, n_classes = Y_train.shape
        self._n_classes = n_classes
        if self._loss_function == 'brier' and self._n_classes != 2:
            sys.exit("Only 2-class problems supported with brier score")

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
        train_labels = np.argmax(Y_train, axis=1).reshape((n_train_items, ))
        if np.max(self._train_proportions) == 1.0:
            self._model = None

        else:
            if self._loss_function == 'log':
                self._model = LogisticRegression(penalty=self._penalty, C=self._alpha, fit_intercept=self._fit_intercept)
            elif self._loss_function == 'brier':
                if self._penalty == 'l1':
                    self._model = Lasso(alpha=self._alpha, fit_intercept=self._fit_intercept)
                elif self._penalty == 'l2':
                    self._model = Ridge(alpha=self._alpha, fit_intercept=self._fit_intercept)
                elif self._penalty is None:
                    self._model = LinearRegression(fit_intercept=self._fit_intercept)
                else:
                    sys.exit('penalty %s not supported with %s loss' % (self._penalty, self._loss_function))
            else:
                sys.exit('Loss function %s not supported' % self._loss_function)
            # train the model using a vector of labels
            self._model.fit(X_train, train_labels, sample_weight=train_weights)

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
        elif self._loss_function == 'log':
            return self._model.predict(X)
        elif self._loss_function == 'brier':
            return np.array(self._model.predict(X) > 0.5, dtype=int)

    def predict_probs(self, X):
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model is None:
            default = np.argmax(self._train_proportions)
            full_probs[:, default] = 1.0
            return full_probs
        elif self._loss_function == 'log':
            # otherwise, get probabilities from the model
            model_probs = self._model.predict_proba(X)
            # map these probabilities back to the full set of classes
            for i, cl in enumerate(self._model.classes_):
                full_probs[:, cl] = model_probs[:, i]
            return full_probs
        elif self._loss_function == 'brier':
            # otherwise, get probabilities from the model
            model_probs = self._model.predict(X)
            # map these probabilities back to the full set of classes
            full_probs[:, 0] = 1 - model_probs
            full_probs[:, 1] = model_probs
            full_probs = np.maximum(full_probs, np.zeros_like(full_probs))
            full_probs = np.minimum(full_probs, np.ones_like(full_probs))
            return full_probs

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
        elif self._loss_function == 'log':
            return self._model.classes_
        elif self._loss_function == 'brier':
            return [0, 1]

    def get_default(self):
        return np.argmax(self._train_proportions)

    def get_col_names(self):
        return self._col_names

    def get_coefs(self, target_class=0):
        coefs = zip(self._col_names, np.zeros(len(self._col_names)))
        if self._model is not None:
            if self._loss_function == 'log':
                for i, cl in enumerate(self._model.classes_):
                    if cl == target_class:
                        coefs = zip(self._col_names, self._model.coef_[i])
                        break
            elif self._loss_function == 'brier':
                coefs = zip(self._col_names, self._model.coef_)
        return coefs

    def get_intercept(self, target_class=0):
        # if we've saved a default value, there are no intercepts
        intercept = 0
        if self._model is not None:
            # otherwise, see if the model an intercept for this class
            if self._loss_function == 'log':
                for i, cl in enumerate(self._model.classes_):
                    if cl == target_class:
                        intercept = self._model.intercept_[i]
                        break
            elif self._loss_function == 'brier':
                intercept = self._model.intercept_
        return intercept

    def get_model_size(self):
        n_nonzeros_coefs = 0
        if self._model is None:
            return 0
        elif self._loss_function == 'log':
            coefs = self._model.coef_
            for coef_list in coefs:
                n_nonzeros_coefs += np.sum([1.0 for c in coef_list if c != 0])
            return n_nonzeros_coefs
        elif self._loss_function == 'brier':
            # TODO implement this
            return 0

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
                  'loss': self._loss_function,
                  'alpha': self.get_alpha(),
                  'penalty': self.get_penalty(),
                  'intercepts': all_intercepts,
                  'coefs': all_coefs,
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  'fit_intercept': self._fit_intercept,
                  'train_f1': self._train_f1,
                  'train_acc': self._train_acc,
                  'dev_f1': self._dev_f1,
                  'dev_acc': self._dev_acc
                  }
        fh.write_to_json(output, os.path.join(self._output_dir, self._name + '_metadata.json'), sort_keys=False)
        fh.write_to_json(self.get_col_names(), os.path.join(self._output_dir, self._name + '_col_names.json'), sort_keys=False)


def load_from_file(model_dir, name):
    input = fh.read_json(os.path.join(model_dir, name + '_metadata.json'))
    col_names = fh.read_json(os.path.join(model_dir, name + '_col_names.json'))
    n_classes = int(input['n_classes'])
    alpha = float(input['alpha'])
    train_proportions = input['train_proportions']
    penalty = input['penalty']
    fit_intercept = input['fit_intercept']
    loss = input['loss']

    classifier = LinearClassifier(alpha, loss, penalty, fit_intercept, output_dir=model_dir, name=name)
    model = joblib.load(os.path.join(model_dir, name + '.pkl'))
    classifier.set_model(model, train_proportions, col_names, n_classes)
    return classifier
