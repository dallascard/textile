import os
from optparse import OptionParser

from core.experiments import combo


def main():
    usage = "%prog project subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--label', dest='label', default='label',
                      help='Label name: default=%default')
    parser.add_option('--field_name', dest='field_name', default='year_group',
                      help='Field to split on: default=%default')
    parser.add_option('--config', dest='config', default='default.json',
                      help='Field to split on: default=%default')
    parser.add_option('-p', dest='calib_prop', default=0.2,
                      help='Proportion of test data to use for calibration: default=%default')
    parser.add_option('-t', dest='train_prop', default=1.0,
                      help='Proportion of training data to use for training: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l2',
                      help='Regularization type: default=%default')
    parser.add_option('-r', dest='repeats', default=3,
                      help='Repeats: default=%default')
    #parser.add_option('-s', dest='size', default=300,
    #                  help='Size of word vectors: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    subset = args[1]

    label = options.label
    field_name = options.field_name
    config_file = os.path.join('config', options.config)
    calib_prop = float(options.calib_prop)
    train_prop = float(options.train_prop)
    penalty = options.penalty
    repeats = int(options.repeats)

    combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                               model_type='LR', label=label, penalty=penalty, repeats=repeats,
                               objective='f1')
    combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                               model_type='LR', label=label, penalty=penalty, repeats=repeats,
                               objective='calibration')
    combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                               model_type='LR', label=label, penalty=penalty, repeats=repeats,
                               objective='f1', cshift='classify')
    combo.cross_train_and_eval(project, subset, field_name, config_file, calib_prop, train_prop, suffix='',
                               model_type='LR', label=label, penalty=penalty, repeats=repeats,
                               objective='calibration', cshift='classify')

if __name__ == '__main__':
    main()
