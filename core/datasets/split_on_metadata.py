import os
import sys
import random
from optparse import OptionParser
from core.util import file_handling as fh


def main():
    usage = "%prog input_json field_name"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='calib_prop', default=0.5,
                      help='Percent to use for the calibration part of each split: default=%default')
    parser.add_option('--sampling', dest='sampling', default='proportional',
                      help='How to divide calibration and test data [proportional|random]: default=%default')
    parser.add_option('-o', action="store_true", dest="overwrite", default=False,
                      help='Overwrite existing files: default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed: default=%default')

    (options, args) = parser.parse_args()
    input_file = args[0]
    field_name = args[1]

    calib_percent = float(options.calib_prop)
    seed = int(options.seed)
    sampling = options.sampling
    overwrite = options.overwrite
    random.seed(seed)

    make_random_split(input_file, field_name, calib_percent, overwrite, sampling)


# MAYBE BETTER JUST TO SPLIT AS NEEDED!

def make_random_split(input_file, field_name, calib_percent, overwrite=False, sampling='proportional'):
    """
    Split a dataset into multiple overlapping datasets based on some metadata variable (such as year)
    The idea is to create subsets to test domain adaptation / covariate shift
    For each value of the variable, create three datasets:
        train = all those items that don't have that value (training data)
        calib = random subset of items that do have that value (calibration data)
        test = remaining items that do have that value (evaluation data)
    :param input_file: 
    :param field_name: 
    :param calib_percent: 
    :param overwrite: 
    :param sampling: 
    :return: 
    """
    basedir = os.path.dirname(input_file)
    data = fh.read_json(input_file)
    field_vals = set([data[k][field_name] for k in data.keys()])

    if sampling == 'proportional':
        for val in field_vals:
            print(val)
            train = {k: v for k, v in data.items() if data[k][field_name] != val}
            subset = {k: v for k, v in data.items() if data[k][field_name] == val}

            keys = list(subset.keys())
            random.shuffle(keys)
            n_items = len(keys)
            print("Loaded %d items" % n_items)

            n_calib = int(n_items * calib_percent)
            calib = {k: data[k] for k in keys[:n_calib]}
            test = {k: data[k] for k in keys[n_calib:]}
            print("Creating train, calibration, and test sets of sizes %d, %d and %d, respectively" % (len(train), len(calib), len(test)))

            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_train.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(train, output_file)

            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_calib.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(calib, output_file)

            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_test.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(test, output_file)

    else:
        keys = list(data.keys())
        random.shuffle(keys)
        n_items = len(keys)
        print("Loaded %d items" % n_items)

        n_calib = int(n_items * calib_percent)
        calib = {k: data[k] for k in keys[:n_calib]}
        test = {k: data[k] for k in keys[n_calib:]}

        for val in field_vals:
            print(val)
            train = {k: v for k, v in data.items() if data[k][field_name] != val}
            calib_subset = {k: v for k, v in calib.items() if calib[k][field_name] == val}
            test_subset = {k: v for k, v in test.items() if test[k][field_name] == val}
            print("Creating train, calibration, and test sets of sizes %d, %d and %d, respectively" % (len(train), len(calib_subset), len(test_subset)))

            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_train.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(train, output_file)

            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_calib.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(calib_subset, output_file)

            output_file = os.path.join(basedir, field_name + '_' + str(val) + '_test.json')
            if os.path.exists(output_file) and not overwrite:
                sys.exit("Error: output file %s exists" % output_file)
            fh.write_to_json(test_subset, output_file)


if __name__ == '__main__':
    main()
