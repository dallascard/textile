import operator
from optparse import OptionParser

from ..models import lr
from ..models import load_model

def main():
    usage = "%prog model_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-n', dest='n_terms', default=10,
                      help='Number of terms to display: default=%default')
    parser.add_option('--model_type', dest='model_type', default=None,
                      help='Model type [LR|MLP|ensemble]; None=auto-detect: default=%default')

    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    model_dir = args[0]

    n_terms = int(options.n_terms)
    model_type = options.model_type
    model = load_model.load_model(model_dir, model_type)

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



if __name__ == '__main__':
    main()
