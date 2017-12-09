from optparse import OptionParser

from ..util.make_slurm_script import make_script


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--offset', dest='offset', default=0,
                      help='Offset: default=%default')
    parser.add_option('--DAN', action="store_true", dest="DAN", default=False,
                      help='Make DAN scripts (instead of LR): default=%default')

    (options, args) = parser.parse_args()

    offset = int(options.offset)
    DAN = options.DAN

    subsets = ['all']
    labels = ['city']
    n_train_vals = [500, 5000]
    use_cshift_options = [False, True]
    objectives = ['f1', 'calibration', 'acc']
    test_years = [2012, 2013, 2014, 2015, 2016]

    if DAN:
        model = 'DAN'
    else:
        model = 'LR'

    seed = 42 + offset
    if model == 'LR':
        hours = 10
        alpha_min = 0.01
        alpha_max = 1000
        n_alphas = 8
    else:
        hours = 48
        alpha_min = 0.0001
        alpha_max = 0.1
        n_alphas = 4
    for subset in subsets:
        for label in labels:
            for objective in objectives:
                for use_cshift in use_cshift_options:
                    for n_train in n_train_vals:
                        for test_year in test_years:
                            cmd = 'python -m core.experiments.over_time_split_and_fit2'
                            cmd += ' projects/yelp/ '
                            if model == 'LR':
                                cmd += ' all config/config.json'
                            else:
                                cmd += ' all config/unigrams.json'
                            cmd += ' --model ' + model
                            cmd += ' --label ' + label
                            cmd += ' --test_start ' + str(test_year)
                            cmd += ' --test_end ' + str(test_year)
                            cmd += ' --n_train ' + str(n_train)
                            if label == 'helpful':
                                cmd += ' --sample'
                            if model == 'LR':
                                if n_train == 500:
                                    cmd += ' --repeats 10'
                                else:
                                    cmd += ' --repeats 5'
                            else:
                                cmd += ' --repeats 1'

                            cmd += ' --n_alphas ' + str(n_alphas)
                            cmd += ' --alpha_min ' + str(alpha_min)
                            cmd += ' --alpha_max ' + str(alpha_max)

                            cmd += ' --objective ' + objective
                            if use_cshift:
                                cmd += ' --cshift'
                                cmd += ' --n_cshift 100000'
                            cmd += ' --seed ' + str(seed)
                            cmd += ' --suffix ' + '_' + str(offset)

                            name = '_'.join(['run', subset, label, model, str(n_train), objective, 'cshift', str(use_cshift), str(test_year)])
                            script = make_script(name, [cmd], 1, 16, hours, False, 'pytorch', [])

                            with open(name + '.sh', 'w') as f:
                                f.write(script)


if __name__ == '__main__':
    main()
