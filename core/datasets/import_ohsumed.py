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

    key = None
    field = None
    text = ''
    type = ''
    title = ''
    terms = None
    count = 0
    for line in train_lines:
        if line.startswith('.I'):
            # save the current article
            if key is not None and terms is not None:
                data[key] = {'text': title + '\n\n' + text, 'type': type, 'year': 1987}
                for term in terms:
                    data[key][term] = {0: 0, 1: 1}
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

    print("Total training articles = %d" % len(data))
    most_common = category_counts.most_common(n=100)
    for t, c in most_common:
        print(t, c)

    for key in data:
        for category in categories:
            if category not in data[key]:
                data[key][category] = {0: 1, 1: 0}

    key = None
    field = None
    text = ''
    type = ''
    title = ''
    terms = None
    count = 0
    for line in test_lines:
        if line.startswith('.I'):
            # save the current article
            if key is not None and terms is not None:
                data[key] = {'text': title + '\n\n' + text, 'type': type, 'year': 1988}
                for term, count in most_common:
                    if term in terms:
                        data[key][term] = {0: 0, 1: 1}
                    else:
                        data[key][term] = {0: 1, 1: 0}
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
            #category_counts.update(terms)
        elif field == 'title':
            title = line.strip()
        elif field == 'type':
            type = line.strip()
        elif field == 'text':
            if text == '':
                text = line.strip()
            else:
                text += '\n\n' + line.strip()

    print("Total articles = %d" % len(data))

    fh.makedirs(dirs.dir_data_raw(project))
    fh.write_to_json(data, os.path.join(dirs.dir_data_raw(project), 'all.json'))


if __name__ == '__main__':
    main()
