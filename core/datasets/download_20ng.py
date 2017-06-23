import os
from optparse import OptionParser

from sklearn.datasets import fetch_20newsgroups
from collections import defaultdict

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog datasets_output_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--subset', dest='subset', default='train',
    #                  help='Subset to download [train|test|all]: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    base_dir = args[0]

    names = ['talk.religion.misc', 'comp.windows.x', 'rec.sport.baseball', 'talk.politics.mideast', 'comp.sys.mac.hardware', 'sci.space', 'talk.politics.guns', 'comp.graphics', 'comp.os.ms-windows.misc', 'soc.religion.christian', 'talk.politics.misc', 'rec.motorcycles', 'comp.sys.ibm.pc.hardware', 'rec.sport.hockey', 'misc.forsale', 'sci.crypt', 'rec.autos', 'sci.med', 'sci.electronics', 'alt.atheism']

    prefixes = set([name.split('.')[0] for name in names])

    groups = {'20ng_all': names}
    for prefix in prefixes:
        name = '20ng_' + prefix
        groups[name] = [cat for cat in names if cat.startswith(prefix)]
        print(prefix, name, groups[name])

    for group in groups.keys():
        if len(groups[group]) > 1:
            for subset in ['train', 'test', 'all']:
                download_articles(base_dir, group, groups[group], subset)


def download_articles(base_dir, name, categories, subset):
    n_categories = len(categories)
    project_dir = os.path.join(base_dir, name)
    fh.makedirs(project_dir)

    data = {}
    newsgroups_data = fetch_20newsgroups(subset=subset, categories=categories, remove=('headers', 'footers', 'quotes'))

    print(len(newsgroups_data['data']))
    #print newsgroups_data['data'][0]

    for i in range(len(newsgroups_data['data'])):
        line = newsgroups_data['data'][i]
        #            line = unicode(line, errors='ignore')
        #unicodedata.normalize('NFKD', line).encode('ascii', 'ignore')
        #line = ''.join(ch for ch in line if unicodedata.category(ch)[0]!="C")
        #contents = line.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace('\0',' ').replace('*', '.').strip().replace('"', "''").replace("\\","\\\\")
        data[str(len(data))] = {'text': line,
                                'label': newsgroups_data['target_names'][newsgroups_data['target'][i]]}

    raw_data_dir = dirs.dir_data_raw(project_dir)
    fh.makedirs(raw_data_dir)
    fh.write_to_json(data, os.path.join(raw_data_dir, subset + '.json'))


if __name__ == '__main__':
    main()
