import os
import sys

import numpy as np

from ..models import linear
from ..util import file_handling as fh


class Ensemble:

    def __init__(self, model_dir, name='ensemble'):
        self._models = {}
        self._model_dir = model_dir
        self._name = name
        self._model_type = 'ensemble'

    def add_model(self, model, name):
        self._models[name] = model

    def get_train_proportions(self):
        prop_list = []
        for model in self._models.values():
            prop_list.append(model.get_train_proportions())
        return np.mean(np.vstack(prop_list), axis=0)

    def predict(self, X):
        pred_probs = self.predict_probs(X)
        return np.argmax(pred_probs, axis=1)

    def predict_probs(self, X, do_platt=False):
        pred_prob_list = []
        for model in self._models.values():
            probs = model.predict_probs(X, do_platt=do_platt)
            pred_prob_list.append(probs)
        tensor = np.array(pred_prob_list)
        return np.mean(tensor, axis=0)

    def predict_proportions(self, X=None, weights=None, do_cfm=False, do_platt=False):
        first_values = []
        second_values = []
        for model in self._models.values():
            first, second = model.predict_proportions(X, weights, do_cfm, do_platt)
            first_values.append(first)
            second_values.append(second)
        return np.mean(first_values, axis=0), np.mean(second_values, axis=0)

    def get_model_type(self):
        return self._model_type

    def get_n_models(self):
        return len(self._models)

    def save(self):
        filename = os.path.join(self._model_dir, self._name + '_metadata.json')
        output = {'model_type': self._model_type,
                  'models': list(self._models.keys())
                  }
        fh.write_to_json(output, filename)


def load_from_file(model_dir, name):
    filename = os.path.join(model_dir, name + '_metadata.json')
    metadata = fh.read_json(filename)
    assert metadata['model_type'] == 'ensemble'
    ensemble = Ensemble(model_dir, name=name)
    models = metadata['models']

    for m in models:
        filename = os.path.join(model_dir, m + '_metadata.json')
        metadata = fh.read_json(filename)
        model_type = metadata['model_type']
        if model_type == 'LR':
            model = linear.load_from_file(model_dir, m)
        #elif model_type == 'MLP':
        #    model = mlp.load_from_file(model_dir, m)
        else:
            sys.exit("Model type %s not recognized" % model_type)
        ensemble.add_model(model, m)

    return ensemble
