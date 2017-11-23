import os
import sys

from ..util import file_handling as fh
from ..models import linear, decision_list, ensemble


def load_model(model_dir, model_name, model_type=None):
    if model_type is None:
        input = fh.read_json(os.path.join(model_dir, model_name + '_metadata.json'))
        if 'model_type' in input:
            model_type = input['model_type']
        else:
            sys.exit("Auto-detect failed; please specify model type")

    if model_type == 'LR':
        return linear.load_from_file(model_dir, model_name)
    #elif model_type == 'MLP':
    #    return mlp.load_from_file(model_dir, model_name)
    elif model_type == 'DL':
        return decision_list.load_from_file(model_dir, model_name)
    elif model_type == 'ensemble':
        return ensemble.load_from_file(model_dir, model_name)
    else:
        sys.exit("Model type not recognized")

