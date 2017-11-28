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

    subsets = ['all']
    labels = ['city']
    n_train_vals = [500, 1000, 2000, 5000]
    use_cshift_options = [False, True]
    objectives = ['f1', 'calibration']
    test_years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017]

    for subset in subsets:
        for label in labels:
            for objective in objectives:
                for use_cshift in use_cshift_options:
                    for n_train in n_train_vals:
                        for test_year in test_years:
                            cmd = 'python -m core.experiments.over_time_split_and_fit'
                            cmd += ' projects/yelp/' + subset
                            cmd += ' all config/default.json'
                            cmd += ' --label ' + label
                            cmd += ' --test_start ' + str(test_year)
                            cmd += ' --test_end ' + str(test_year)
                            cmd += ' --n_train ' + str(n_train)
                            if label == 'helpful':
                                cmd += ' --sample'
                            if n_train_vals < 800:
                                cmd += ' --repeats 20'
                            if n_train_vals < 1500:
                                cmd += ' --repeats 10'
                            else:
                                cmd += ' --repeats 5'
                            cmd += ' --objective ' + objective
                            if use_cshift:
                                cmd += ' --cshift'

                            name = '_'.join(['run', subset, label, str(n_train), objective, 'cshift', str(use_cshift), str(test_year)])
                            script = make_script(name, [cmd], 1, 1, 10, False, 'pytorch', [])

                            with open(name + '.sh', 'w') as f:
                                f.write(script)


if __name__ == '__main__':
    main()
