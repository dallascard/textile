import operator
from optparse import OptionParser

from ..models import lr

def main():
    usage = "%prog model_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    model_dir = args[0]

    model = lr.load_from_file(model_dir)
    classes = model.get_active_classes()
    for c in classes:
        coefs = model.get_coefs(target_class=c)
        coefs_sorted = sorted(coefs, key=operator.itemgetter(1))
        terms, values = zip(*coefs_sorted)
        output = str(c) + ': ' + ' '.join([t for t in terms[-1:-10:-1]])
        print(output)

if __name__ == '__main__':
    main()
