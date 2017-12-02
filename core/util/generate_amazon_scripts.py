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

    #subsets = ['clothes5', 'home5', 'sports5', 'video5']
    subsets = ['toys5', 'tools5', 'clothes5', 'sports5']
    labels = ['helpfulness', 'fivestar']
    n_train_vals = [500, 5000]
    use_cshift_options = [False, True]
    objectives = ['f1', 'calibration']

    seed = 42
    for objective in objectives:
        for use_cshift in use_cshift_options:
            for subset in subsets:
                for label in labels:
                    for n_train in n_train_vals:
                        cmd = 'python -m core.experiments.over_time_split_and_fit'
                        cmd += ' projects/amazon/' + subset
                        cmd += ' all config/n5grams.json'
                        cmd += ' --label ' + label
                        cmd += ' --test_start 2013'
                        cmd += ' --test_end 2014'
                        cmd += ' --n_train ' + str(n_train)
                        cmd += ' --sample'
                        if n_train == 500:
                            cmd += ' --repeats 10'
                        else:
                            cmd += ' --repeats 5'
                        cmd += ' --objective ' + objective
                        if use_cshift:
                            cmd += ' --cshift'
                        cmd += ' --seed ' + str(seed)

                        name = '_'.join(['run', subset, label, str(n_train), objective, 'cshift', str(use_cshift)])
                        script = make_script(name, [cmd], 1, 16, 10, False, 'pytorch', [])

                        with open(name + '.sh', 'w') as f:
                            f.write(script)



if __name__ == '__main__':
    main()
