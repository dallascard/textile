import os
import operator

import numpy as np

from ..util import file_handling as fh
from ..models import blr_fit


class BLR:
    """
    Bayesian logistic regression model (including Automatic Relevance Determination)
    Currently only binary labels are supported
    """
    def __init__(self, alpha=1.0, s_0=1e-2, r_0=1e-4, fit_intercept=True, n_classes=2):
        self._model_type = 'BLR'
        self._alpha = alpha
        self._s_0 = s_0
        self._r_0 = r_0
        self._fit_intercept = fit_intercept
        if n_classes != 2:
            print("Only binary labels are currently supported")
            raise Exception
        self._n_classes = n_classes

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the column names of the feature matrix
        self._col_names = None
        # create variables to hold the estimated parameters
        self._m = None
        self._V = None
        self._inv_V = None

    def get_model_type(self):
        return self._model_type

    def set_model(self, train_proportions, col_names, m, V, inv_V):
        self._col_names = col_names
        self._train_proportions = train_proportions
        self._m = m
        self._V = V
        self._inv_V = inv_V

    def fit(self, X, y, col_names, sample_weights=None, batch=True, multilevel=True, ard=True, max_iter=500, tol=1e-6):

        # store the proportion of class labels in the training data
        bincount = np.bincount(np.array(y, dtype=int), minlength=self._n_classes)
        self._train_proportions = (bincount / float(bincount.sum())).tolist()
        self._col_names = col_names

        if batch:
            if multilevel:
                if ard:
                    m, V, inv_V, E_alphas = blr_fit.batch_multilevel_ard(X, y, self._fit_intercept, sample_weights, self._s_0, self._r_0, max_iter, tol)
                else:
                    m, V, inv_V, E_alpha = blr_fit.batch_multilevel(X, y, self._fit_intercept, sample_weights, self._s_0, self._r_0, max_iter, tol)
            else:
                m, V, inv_V = blr_fit.batch_fixed_alpha(X, y, self._alpha, self._fit_intercept, sample_weights=sample_weights, max_iter=max_iter, tol=tol)
        else:
            m, V, inv_V = blr_fit.iterative_fixed_alpha(X, y, self._alpha, self._fit_intercept, sample_weights, max_iter, tol)

        self._m = m
        self._V = V
        self._inv_V = inv_V

    def predict(self, X, batch=True, sampling=True, max_iter=500, tol=1e-6):
        if self._m is None:
            return None
        else:
            p_y_given_x = self.predict_probs(X, batch, sampling, max_iter, tol)
            return p_y_given_x > 0.5

    def predict_probs(self, X, batch=True, sampling=True, max_iter=500, tol=1e-6):
        # if we've stored a default value, then that is our prediction
        if self._m is None:
            return None
        else:
            if sampling:
                p_y_given_x = np.mean(blr_fit.sample_predictions(X, self._m, self._V, self._fit_intercept), axis=1)

            else:
                if batch:
                    p_y_given_x = blr_fit.batch_predictive_density(X, self._m, self._V, self._inv_V, self._fit_intercept, max_iter, tol)
                else:
                    p_y_given_x = blr_fit.iterative_predictive_density(X, self._m, self._V, self._inv_V, self._fit_intercept, max_iter, tol)
            return p_y_given_x

    def sample_probs(self, X, n_samples=20):
        # if we've stored a default value, then that is our prediction
        if self._m is None:
            return None
        else:
            p_y_given_x = blr_fit.sample_predictions(X, self._m, self._V, self._fit_intercept, n_samples=n_samples)
            return p_y_given_x

    def get_n_classes(self):
        return self._n_classes

    def get_train_proportions(self):
        return self._train_proportions

    def get_default(self):
        return np.argmax(self._train_proportions)

    def get_col_names(self):
        return self._col_names

    def get_coefs(self, target_class=1):
        """Return the mean of the approximate normal posterior over weights"""
        if self._m is not None:
            if self._fit_intercept:
                return zip(self._col_names, self._m[1:] * (target_class*2-1))
            else:
                return zip(self._col_names, self._m[1:] * (target_class*2-1))
        else:
            return None

    def get_intercept(self, target_class=1):
        """Return the mean of the approximate normal posterior over the intercept (if present)"""
        if self._m is not None:
            if self._fit_intercept:
                return self._m[0] * (target_class*2-1)
            else:
                return 0 * (target_class*2-1)
        else:
            return None

    def get_model_size(self):
        return len(self._m)

    def get_active_classes(self):
        if self._m is None:
            return None
        else:
            return np.nonzero(self._train_proportions)[0]

    def save(self, output_dir):
        print("Saving model")
        np.savez(os.path.join(output_dir, 'model.npz'), m=self._m, V=self._V, inv_V=self._inv_V)

        output = {'model_type': 'LR',
                  'alpha': self._alpha,
                  'fit_intercept': self._fit_intercept,
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  's_0': self._s_0,
                  'r_0': self._r_0
        }

        if self._m is not None:
            coefs_dict = self.get_coefs()
            output['coefs_mean'] = sorted(coefs_dict.items(), key=operator.itemgetter(1))
            output['intercept_mean'] = self.get_intercept()

        fh.write_to_json(output, os.path.join(output_dir, 'metadata.json'), sort_keys=False)
        fh.write_to_json(self.get_col_names(), os.path.join(output_dir, 'col_names.json'), sort_keys=False)


def load_from_file(model_dir):
    input = fh.read_json(os.path.join(model_dir, 'metadata.json'))
    col_names = fh.read_json(os.path.join(model_dir, 'col_names.json'))
    n_classes = int(input['n_classes'])
    alpha = input['alpha']
    train_proportions = input['train_proportions']
    fit_intercept = input['fit_intercept']
    s_0 = input['s_0']
    r_0 = input['r_0']

    classifier = BLR(alpha, s_0, r_0, fit_intercept, n_classes)
    params = np.load(os.path.join(model_dir, 'model.npz'))
    classifier.set_model(train_proportions, col_names, params['m'], params['V'], params['inv_V'])
    return classifier
