import os


def dir_data(project):
    return os.path.join(project, 'data')


def dir_data_raw(project):
    return os.path.join(project, 'data', 'raw')


def dir_subsets(project):
    return os.path.join(project, 'data', 'subsets')


def dir_subset(project, subset):
    return os.path.join(project, 'data', 'subsets', subset)


def dir_labels(project, subset):
    return os.path.join(project, 'data', 'subsets', subset, 'labels')


def dir_features(project, subset):
    return os.path.join(project, 'data', 'subsets', subset, 'features')


def dir_weights(project, subset):
    return os.path.join(project, 'data', 'subsets', subset, 'weights')


def dir_predictions(project, subset, model):
    return os.path.join(project, 'data', 'subsets', subset, 'predictions', model)


def dir_patterns(project, subset):
    return os.path.join(project, 'data', 'subsets', subset, 'patterns')


def dir_partitions(project, subset):
    return os.path.join(project, 'data', 'subsets', subset, 'partitions')


def dir_models(project):
    return os.path.join(project, 'models')


def dir_export(project):
    return os.path.join(project, 'export')