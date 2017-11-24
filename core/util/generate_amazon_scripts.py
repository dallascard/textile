from optparse import OptionParser

from ..util.make_slurm_script import make_script


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    subsets = ['clothes5', 'home5', 'sports5', 'video5']
    labels = ['helpfulness', 'fivestar']
    n_train_vals = [1000, 2000, 5000, 10000, 20000]
    use_cshift_options = [False, True]
    objectives = ['f1', 'calibration']

    for subset in subsets:
        for label in labels:
            for objective in objectives:
                for use_cshift in use_cshift_options:
                    for n_train in n_train_vals:
                        cmd = 'python -m core.experiment.over_time_split_and_fit'
                        cmd += ' projects/amazon/' + subset
                        cmd += ' all config/n5grams.json'
                        cmd += ' --label ' + label
                        cmd += ' --test_start 2013'
                        cmd += ' --test_end 2014'
                        cmd += ' --n_train ' + str(n_train)
                        cmd += ' --sample'
                        cmd += ' --repeats 5'
                        cmd += ' --objective ' + objective
                        if use_cshift:
                            cmd += ' --cshift'

                        name = '_'.join(['run', subset, label, str(n_train), objective, 'cshift', str(use_cshift)])
                        script = make_script(name, cmd, 1, 1, 10, False, 'pytorch', [])

                        with open(name + '.sh', 'w') as f:
                            f.write(script)


if __name__ == '__main__':
    main()
