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

    #subsets = ['clothes5', 'home5', 'sports5', 'video5']
    subsets = ['toys5', 'tools5', 'clothes5', 'sports5']
    labels = ['helpfulness', 'fivestar']
    n_train_vals = [500, 5000]
    use_cshift_options = [False, True]
    objectives = ['f1', 'calibration', 'acc']

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
    for objective in objectives:
        for use_cshift in use_cshift_options:
            for subset in subsets:
                for label in labels:
                    for n_train in n_train_vals:
                        cmd = 'python -m core.experiments.over_time_split_and_fit'
                        cmd += ' projects/amazon/' + subset
                        if model == 'LR':
                            cmd += ' all config/n5grams.json'
                        else:
                            cmd += ' all config/unigrams.json'
                        cmd += ' --model ' + model
                        cmd += ' --label ' + label
                        cmd += ' --test_start 2013'
                        cmd += ' --test_end 2014'
                        cmd += ' --n_train ' + str(n_train)
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
                        cmd += ' --seed ' + str(seed)
                        cmd += ' --suffix ' + '_' + str(offset)

                        name = '_'.join(['run', subset, label, model, str(n_train), objective, 'cshift', str(use_cshift)])
                        script = make_script(name, [cmd], 1, 16, hours, False, 'pytorch', [])

                        with open(name + '.sh', 'w') as f:
                            f.write(script)



if __name__ == '__main__':
    main()
