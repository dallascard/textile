import os
import operator

import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression as lr

from ..util import file_handling as fh

class MLP:
    """
    Multilayer perceptron (representing documents as weighted sums of word vectors)
    """
    def __init__(self, alpha=1e-3, penalty=None, fit_intercept=True, n_classes=2, n_layers=1):
        self._model_type = 'MLP'
        self._alpha = alpha
        self._penalty = penalty
        self._fit_intercept = fit_intercept
        self._n_classes = n_classes
        self.n_layers = n_layers

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
    def fit(self, X, Y, col_names, sample_weights=None):
        """
        Fit a classifier to data
        :param X: feature matrix: np.array(size=(n_items, n_features))
        :param Y: one-hot label encoding np.array(size=(n_items, n_classes))
        :return: None
        """
        # store the proportion of class labels in the training data
        class_sums = np.array(np.sum(Y, axis=0), dtype=float)
        self._train_proportions = (class_sums / np.sum(class_sums)).tolist()
        self._col_names = col_names

        # if there is only a single type of label, make a default prediction
        if np.max(self._train_proportions) == 1.0:
            self._model = None
        else:
            self._model = lr(penalty=self._penalty, C=self._alpha, fit_intercept=self._fit_intercept)
            # otherwise, train the model
            self._model.fit(X, y, sample_weight=sample_weights)

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
            model_probs = self._model.predict_proba(X)
            # map these probabilities back to the full set of classes
            for i, cl in enumerate(self._model.classes_):
                full_probs[:, cl] = model_probs[:, i]
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
            # otherwise, see if the model an intercept for this class
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


def load_from_file(model_dir):
    input = fh.read_json(os.path.join(model_dir, 'metadata.json'))
    col_names = fh.read_json(os.path.join(model_dir, 'col_names.json'))
    n_classes = int(input['n_classes'])
    alpha = float(input['alpha'])
    train_proportions = input['train_proportions']
    penalty = input['penalty']
    fit_intercept = input['fit_intercept']

    classifier = LR(alpha, penalty, fit_intercept, n_classes)
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    classifier.set_model(model, train_proportions, col_names)
    return classifier
