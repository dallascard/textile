import os
from optparse import OptionParser

from core.experiments import combo


def main():
    usage = "%prog project"
    parser = OptionParser(usage=usage)
    parser.add_option('--field_name', dest='field_name', default='year_group',
                      help='Field to split on: default=%default')
    parser.add_option('--config', dest='config', default='default.json',
                      help='Field to split on: default=%default')
    parser.add_option('--n_train', dest='n_train', default=100,
                      help='Number of training instances to use (0 for all): default=%default')
    parser.add_option('--n_calib', dest='n_calib', default=100,
                      help='Number of test instances to use for calibration: default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels instead of averaging: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type: default=%default')
    parser.add_option('-r', dest='repeats', default=3,
                      help='Repeats: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|MLP]: default=%default')


    (options, args) = parser.parse_args()
    project = args[0]

    field_name = options.field_name
    config_file = os.path.join('config', options.config)
    n_train = int(options.n_train)
    n_calib = int(options.n_calib)
    sample_labels = options.sample
    penalty = options.penalty
    repeats = int(options.repeats)
    model_type = options.model

    pairs = [('pro_tone', 'label'), ('framing', 'Economic'), ('framing', 'Legality'), ('framing', 'Health'), ('framing', 'Political')]

    for subset, label in pairs:
        print("\n\nStarting", subset, label)
        combo.cross_train_and_eval(project, subset, field_name, config_file, n_calib, n_train, suffix='',
                                   model_type=model_type, label=label, penalty=penalty, repeats=repeats,
                                   objective='f1', sample_labels=sample_labels)
        combo.cross_train_and_eval(project, subset, field_name, config_file, n_calib, n_train, suffix='',
                                   model_type=model_type, label=label, penalty=penalty, repeats=repeats,
                                   objective='calibration', sample_labels=sample_labels)
        #combo.cross_train_and_eval(project, subset, field_name, config_file, n_calib, n_train, suffix='',
        #                           model_type=model_type, label=label, penalty=penalty, repeats=repeats,
        #                           objective='f1', cshift='classify', sample_labels=sample_labels)
        #combo.cross_train_and_eval(project, subset, field_name, config_file, n_calib, n_train, suffix='',
        #                           model_type=model_type, label=label, penalty=penalty, repeats=repeats,
        #                           objective='calibration', cshift='classify', sample_labels=sample_labels)

if __name__ == '__main__':
    main()
