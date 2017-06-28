import os
import sys
import random
from optparse import OptionParser
from ..util import file_handling as fh

def main():
    usage = "%prog input_json field_name"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='train_percent', default=0.5,
                      help='Percent to use for training for each split: default=%default')
    parser.add_option('-s', dest='seed', default=42,
                      help='Random seed: default=%default')
    parser.add_option('-o', action="store_true", dest="overwrite", default=False,
                      help='Overwrite existing files: default=%default')

    (options, args) = parser.parse_args()
    input_file = args[0]
    field_name = args[1]

    train_percent = float(options.train_percent)
    seed = int(options.seed)
    overwrite = options.overwrite
    random.seed(seed)

    make_random_split(input_file, field_name, train_percent, overwrite)


def make_random_split(input_file, field_name, train_percent, overwrite=False):
    basedir = os.path.dirname(input_file)
    data = fh.read_json(input_file)
    field_vals = set([data[k][field_name] for k in data.keys()])
    for val in field_vals:
        print(val)
        subset = {k: v for k, v in data.items() if data[k][field_name] == val}

        keys = list(subset.keys())
        random.shuffle(keys)
        n_items = len(keys)
        print("Loaded %d items" % n_items)

        n_train = int(n_items * train_percent)
        train = {k: data[k] for k in keys[:n_train]}
        other = {k: data[k] for k in keys[n_train:]}
        print("Creating train and dev sets of sizes %d and %d, respectively" % (len(train), len(other)))

        output_file = os.path.join(basedir, field_name + '_' + str(val) + '_train.json')
        if os.path.exists(output_file) and not overwrite:
            sys.exit("Error: output file %s exists" % output_file)
        fh.write_to_json(train, output_file)

        if train_percent < 1.0:
            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_dev.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(other, output_file)


if __name__ == '__main__':
    main()
