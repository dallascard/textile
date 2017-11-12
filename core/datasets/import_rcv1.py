import codecs
import zipfile
from optparse import OptionParser


def main():
    usage = "%prog file.zip"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    filename = args[0]


def process_articles(filename):

    with zipfile.ZipFile(filename, 'r') as z:
        names = z.namelist()
        for name in names:
            with codecs.open(name, 'r', encoding='utf-8') as f:
                text = f.read()



if __name__ == '__main__':
    main()
