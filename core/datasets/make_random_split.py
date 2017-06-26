import os
import sys
import random
from optparse import OptionParser
from ..util import file_handling as fh

d
def main():
    usage = "%prog input_json output_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='train_percent', default=0.5,
                      help='Percent to use for training part: default=%default')
    parser.add_option('-s', dest='seed', default=42,
                      help='Random seed: default=%default')
    parser.add_option('-o', action="store_true", dest="overwrite", default=False,
                      help='Overwrite existing files: default=%default')

    (options, args) = parser.parse_args()
    input_file = args[0]
    output_prefix = args[1]

    train_percent = float(options.train_percent)
    seed = int(options.seed)
    overwrite = options.overwrite
    random.seed(seed)

    make_random_split(input_file, output_prefix, train_percent, overwrite)


def make_random_split(input_file, output_prefix, train_percent, overwrite=False):
    basedir = os.path.dirname(input_file)
    data = fh.read_json(input_file)
    keys = list(data.keys())
    random.shuffle(keys)
    n_items = len(keys)
    n_train = int(n_items * train_percent)

    train = {k: data[k] for k in keys[:n_train]}
    other = {k: data[k] for k in keys[n_train:]}

    output_file = os.path.join(basedir, output_prefix + '_train.json')
    if os.path.exists(output_file) and not overwrite:
        sys.exit("Error: output file %s exists" % output_file)
    fh.write_to_json(train, output_file)

    output_file = os.path.join(basedir, output_prefix + '_dev.json')
    if os.path.exists(output_file) and not overwrite:
        sys.exit("Error: output file %s exists" % output_file)
    fh.write_to_json(other, output_file)


if __name__ == '__main__':
    main()
