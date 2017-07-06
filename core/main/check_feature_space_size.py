from optparse import OptionParser

from ..util import file_handling as fh
from ..preprocessing import features
from ..util import dirs


def main():
    usage = "%prog project_dir subset config.json"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    config_file = args[2]

    config = fh.read_json(config_file)
    feature_defs = []
    for f in config['feature_defs']:
        feature_defs.append(features.parse_feature_string(f))

    check_size(project_dir, subset, feature_defs)


def check_size(project_dir, subset, feature_defs):

    features_dir = dirs.dir_features(project_dir, subset)

    print("loading features")
    feature_list = []
    for feature_def in feature_defs:
        print(feature_def)
        name = feature_def.name
        feature = features.load_from_file(input_dir=features_dir, basename=name)
        feature.threshold(feature_def.min_df)
        feature.transform(feature_def.transform)
        feature_list.append(feature)

    features_concat = features.concatenate(feature_list)
    col_names = features_concat.terms
    print("dimension = %d" % len(col_names))

if __name__ == '__main__':
    main()
