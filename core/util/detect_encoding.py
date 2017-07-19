import chardet
from optparse import OptionParser


def main():
    usage = "%prog filename"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    filename = args[0]
    encoding = detect_encoding(filename)


def detect_encoding(filename):
    """
    Attempt to detect the encoding of a file; note that this can also be done on the command line using:
    > chardetect filename
    :param filename: the name of the file
    :return: the estimaetd encoding
    """
    with open(filename, 'rb') as input_file:
        raw = input_file.read()
    detected_encoding = chardet.detect(raw)
    print("Detected %s with %0.2f confidence" % (detected_encoding['encoding'], detected_encoding['confidence']))
    encoding = detected_encoding['encoding']
    return encoding


if __name__ == '__main__':
    main()
