import os
import glob
from optparse import OptionParser
from collections import defaultdict, Counter

from ..util import dirs
from ..util import file_handling as fh
from ..preprocessing import normalize_text


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

    parties = Counter()
    output = {}
    files = glob.glob(os.path.join(input_dir, '*.json'))
    articles = []
    for f in files:
        data = fh.read_json(f)
        articles.extend(data)

    for i, article in enumerate(articles):
        party_id = article['speaker']['party']['id']
        party_name = article['speaker']['party']['party']
        parties.update([party_name])

    for key, value in parties.most_common():
        print('%s: %d' % (key, value))

    for i, article in enumerate(articles):
        key = article['id']
        text = article['ruling_comments']
        text = normalize_text.strip_html(text)
        text = normalize_text.fix_web_text(text)
        year = article['ruling_date'][:4]
        party_name = article['speaker']['party']['party']
        if party_name == 'Republican':
            output[key] = {'text': text, 'labels': {'republican': {1: 1}}, 'year': int(year)}
        elif party_name == 'Democrat':
            output[key] = {'text': text, 'labels': {'republican': {0: 1}}, 'year': int(year)}

    print("Saving %d articles" % len(output))
    fh.makedirs(dirs.dir_data_raw(project))
    output_file = os.path.join(dirs.dir_data_raw(project), 'all.json')
    fh.write_to_json(output, output_filename=output_file)


if __name__ == '__main__':
    main()
