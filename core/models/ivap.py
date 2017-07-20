from optparse import OptionParser


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()


def estimate_probs_brute_force(model, calib_X, calib_y, test_X):
    calib_pred_probs = model.predict_probs(calib_X)
    n_items, n_classes = calib_pred_probs.shape
    assert n_classes == 2

    test_pred_probs = model.predict_probs(test_X)





if __name__ == '__main__':
    main()
