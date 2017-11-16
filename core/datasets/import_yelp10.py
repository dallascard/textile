import os
from collections import Counter
from optparse import OptionParser


from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog input_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]

    cities = Counter()

    lines = fh.read_json_lines(os.path.join(input_dir, 'business.json'))
    for line in lines:
        city = line['city']
        cities.update([city])

    n = 20
    most_common = cities.most_common(n)
    for i in range(n):
        print(most_common[i])


if __name__ == '__main__':
    main()
