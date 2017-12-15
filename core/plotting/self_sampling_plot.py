from optparse import OptionParser

import matplotlib.pylot as plt
import numpy as np



def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    #pcc = [0.0602, 0.0466, 0.0212, 0.0127, 0.0077, 0.0036]
    #srs = [0.0414, 0.0281, 0.0177, 0.0127, ]


if __name__ == '__main__':
    main()
