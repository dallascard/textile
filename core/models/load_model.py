import os
import sys

from ..util import dirs
from ..util import file_handling as fh
from ..models import lr, blr


def load_model(model_dir, model_type=None):
    if model_type is None:
        input = fh.read_json(os.path.join(model_dir, 'metadata.json'))
        if 'model_type' in input:
            if input['model_type'] == 'LR':
                return lr.load_from_file(model_dir)
            elif input['model_type'] == 'BLR':
                return blr.load_from_file(model_dir)
        else:
            sys.exit("Auto-detect failed; please specify model type")
    elif model_type == 'LR':
        return lr.load_from_file(model_dir)
    elif model_type == 'BLR':
        return blr.load_from_file(model_dir)
    else:
        sys.exit("Model type not recognized")

