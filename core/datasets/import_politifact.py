import os
import glob
from optparse import OptionParser
from collections import defaultdict

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog input_dir project_name"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    project = args[1]

    import_politifact_data(input_dir, project)


def import_politifact_data(input_dir, project):

    parties = defaultdict(set)
    output = {}
    files = glob.glob(os.path.join(input_dir, '*'))
    articles = []
    for f in files:
        data = fh.read_json(f)
        articles.extend(data)

    for i, article in enumerate(articles):
        party_id = article['speaker']['party']['id']
        party_name = article['speaker']['party']['party']
        parties[party_id].add(party_name)

    print(parties)

    for i, article in enumerate(articles):
        key = article['id']
        text = article['ruling_comments']
        party_name = article['speaker']['party']['party']
        if party_name == 'Republican':
            output[key] = {'text': text, 'republican': {1: 1}}
        elif party_name == 'Democrat':
            output[key] = {'text': text, 'republican': {0: 1}}

    print("Saving %d articles" % len(output))
    fh.makedirs(dirs.dir_data_raw(project))
    output_file = os.path.join(dirs.dir_data_raw(project), 'all.json')
    fh.write_to_json(output, output_filename=output_file)


if __name__ == '__main__':
    main()
