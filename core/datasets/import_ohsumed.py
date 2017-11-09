import os
from collections import Counter
from optparse import OptionParser

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog project_dir train_file test_file"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    train_file = args[1]
    test_file = args[2]

    train_lines = fh.read_text(train_file)
    test_lines = fh.read_text(test_file)

    categories = {}
    category_counts = Counter()
    data = {}

    data, categories, category_counts = parse_data(1987, train_lines, data, categories, category_counts, update_counts=True)
    print("Total training articles = %d" % len(data))

    data, categories, category_counts = parse_data(1988, test_lines, data, categories, category_counts, update_counts=False)
    print("Total articles = %d" % len(data))

    most_common = category_counts.most_common(n=100)
    for t, c in most_common:
        print(t, c)

    """
    for k_i, key in enumerate(data):
        for category in categories:
            if category not in data[key]:
                data[key][category] = {0: 1, 1: 0}
        if k_i % 10000 == 0:
            print(k_i)
    """

    fh.makedirs(dirs.dir_data_raw(project))
    fh.write_to_json(data, os.path.join(dirs.dir_data_raw(project), 'all.json'), sort_keys=True)
    fh.write_to_json(most_common, os.path.join(dirs.dir_data_raw(project), 'most_common_categories.json'), sort_keys=False)


def parse_data(year, lines, data, categories, category_counts, update_counts=False):
    key = None
    field = None
    text = ''
    type = ''
    title = ''
    terms = None
    count = 0
    for line in lines:
        if line.startswith('.I'):
            # save the current article
            if key is not None and terms is not None:
                data[key] = {'text': title + '\n\n' + text, 'type': type, 'year': year}
                for term in terms:
                    data[key][term] = 1
            # go on to the next article
            key = line.split()[1].strip()
            text = ''
            count += 1
            if count % 10000 == 0:
                print(count)
        elif line.startswith('.U'):
            field = None
        elif line.startswith('.S'):
            field = None
        elif line.startswith('.M'):
            field = 'terms'
        elif line.startswith('.T'):
            field = 'title'
        elif line.startswith('.P'):
            field = 'type'
        elif line.startswith('.W'):
            field = 'text'
        elif line.startswith('.A'):
            field = None
        elif field == 'terms':
            terms = line.strip().split(';')
            terms = [term.strip() for term in terms]
            for term in terms:
                if term not in categories:
                    categories[term] = len(categories)
            if update_counts:
                category_counts.update(terms)
        elif field == 'title':
            title = line.strip()
        elif field == 'type':
            type = line.strip()
        elif field == 'text':
            if text == '':
                text = line.strip()
            else:
                text += '\n\n' + line.strip()

    return data, categories, category_counts


if __name__ == '__main__':
    main()
