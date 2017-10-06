import os
import glob
import operator
from optparse import OptionParser
from collections import defaultdict

import numpy as np

from ..models import load_model
from ..util import file_handling as fh

def main():
    usage = "%prog model_dir "
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n_terms', default=10,
                      help='Number of terms to display: default=%default')
    #parser.add_option('--model_type', dest='model_type', default=None,
    #                  help='Model type [LR|MLP|ensemble]; None=auto-detect: default=%default')
    #parser.add_option('--values', action="store_true", dest="values", default=False,
    #                  help='Print values: default=%default')

    (options, args) = parser.parse_args()
    model_dir = args[0]

    n_terms = int(options.n_terms)

    top_features = get_top_features(model_dir, n_terms)
    for feature, weight in top_features:
        print('%s\t%0.4f' % (feature, weight))


def get_top_features(model_dir, n_terms, default_model_type=None):

    basename = os.path.split(model_dir)[-1]
    model_files = glob.glob(os.path.join(model_dir, basename + '*_metadata.json'))

    totals = defaultdict(float)

    for file_i, file in enumerate(model_files):
        print("Loading %s" % file)
        model_name = os.path.split(file)[-1][:-14]
        print(model_name)
        model = load_model.load_model(model_dir, model_name, default_model_type)
        model_type = model.get_model_type()
        print("Found: ", model_type)

        if model_type == 'LR':
            classes = model.get_active_classes()
            if len(classes) == 2:
                coefs = model.get_coefs(target_class=0)
                for coef, value in coefs:
                    totals[coef] += value

    coef_totals = [(coef, value) for coef, value in totals.items()]
    coef_totals = sorted(coef_totals, key=lambda x: abs(x[1]))
    coef_totals.reverse()

    return coef_totals[:n_terms]


if __name__ == '__main__':
    main()
