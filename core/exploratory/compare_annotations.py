from optparse import OptionParser


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()


"""
Look at the individual words actually highlighted by annotators in one time as compared to another, 
and see to what extent these explain differences in annotated proportions.
"""


if __name__ == '__main__':
    main()


