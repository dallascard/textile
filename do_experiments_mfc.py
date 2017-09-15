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
    parser.add_option('-p', dest='calib_prop', default=0.1,
                      help='Proportion of test data to use for calibration: default=%default')
    parser.add_option('-t', dest='train_prop', default=0.9,
                      help='Proportion of training data to use for training: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type: default=%default')
    parser.add_option('-r', dest='repeats', default=5,
                      help='Repeats: default=%default')
    parser.add_option('--model', dest='model', default='LR',
                      help='Model type [LR|MLP]: default=%default')


    (options, args) = parser.parse_args()
    project = args[0]

    field_name = options.field_name
    config_file = os.path.join('config', options.config)
    calib_prop = float(options.calib_prop)
    train_prop = float(options.train_prop)
    penalty = options.penalty
    repeats = int(options.repeats)
    model_type = options.model

    pairs = [('pro_tone', 'label'), ('framing', 'Economic'), ('framing', 'Legality'), ('framing', 'Health'), ('framing', 'Political')]

    for subset, label in pairs:
        print("\n\nStarting", subset, label)
        combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                                   model_type=model_type, label=label, penalty=penalty, repeats=repeats,
                                   objective='f1')
        combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                                   model_type=model_type, label=label, penalty=penalty, repeats=repeats,
                                   objective='calibration')
        combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                                   model_type=model_type, label=label, penalty=penalty, repeats=repeats,
                                   objective='f1', cshift='classify')
        combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                                   model_type=model_type, label=label, penalty=penalty, repeats=repeats,
                                   objective='calibration', cshift='classify')

if __name__ == '__main__':
    main()
