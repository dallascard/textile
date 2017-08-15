import os
import operator
from optparse import OptionParser

import numpy as np

from ..models import lr
from ..models import load_model, mlp
from ..util import file_handling as fh

def main():
    usage = "%prog model_dir model_name"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n_terms', default=10,
                      help='Number of terms to display: default=%default')
    parser.add_option('--model_type', dest='model_type', default=None,
                      help='Model type [LR|MLP|ensemble]; None=auto-detect: default=%default')

    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    model_dir = args[0]
    model_name = args[1]

    n_terms = int(options.n_terms)
    model_type = options.model_type
    model = load_model.load_model(model_dir, model_name, model_type)
    model_type = model.get_model_type()

    if model_type == 'LR':
        classes = model.get_active_classes()
        if len(classes) == 2:
            coefs = model.get_coefs(target_class=0)
            coefs_sorted = sorted(coefs, key=operator.itemgetter(1))
            terms, values = zip(*coefs_sorted)
            output = str(0) + ': ' + ' '.join([t for t in terms[-1:-n_terms:-1]])
            print(output)
            output = str(1) + ': ' + ' '.join([t for t in terms[:n_terms]])
            print(output)
        else:
            for c in classes:
                coefs = model.get_coefs(target_class=c)
                coefs_sorted = sorted(coefs, key=operator.itemgetter(1))
                terms, values = zip(*coefs_sorted)
                output = str(c) + ': ' + ' '.join([t for t in terms[-1:-n_terms:-1]])
                print(output)

    elif model_type == 'MLP':
        features_file = os.path.join(model_dir, 'features.json')
        features = fh.read_json(features_file)
        word_vectors_prefix = features[0]['word_vectors_prefix']
        word_vectors = fh.load_dense(word_vectors_prefix + '.npz')
        word_vector_terms = fh.read_json(word_vectors_prefix + '.json')
        n_classes = model.get_n_classes()

        activations = model.predict_probs(word_vectors)
        print(activations.min(), activations.mean(), activations.max())
        print(word_vectors.shape)
        for cl in range(n_classes):
            order = np.argsort(activations[:, cl]).tolist()
            order.reverse()
            terms = [word_vector_terms[i] for i in order[:n_terms]]
            output = str(cl) + ': ' + ' '.join(terms)
            print(output)



if __name__ == '__main__':
    main()
