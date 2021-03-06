
import os
from optparse import OptionParser

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog input_dir project_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    project_dir = args[1]

    city_lookup = dict()

    print("Reading in business data")
    lines = fh.read_json_lines(os.path.join(input_dir, 'business.json'))
    for key, line in lines.items():
        city = line['city']
        business_id = line['business_id']
        city_lookup[business_id] = city

    data = {}

    print("Reading in review data")
    lines = fh.read_json_lines(os.path.join(input_dir, 'review.json'))
    toronto_count = 0
    phoenix_count = 0
    for key, line in lines.items():
        if key % 10000 == 0:
            print(key)
        review_id = line['review_id']
        text = line['text']
        date = line['date']
        year = date.split('-')[0]
        funny = int(line['funny'])
        useful = int(line['useful'])
        cool = int(line['cool'])
        business_id = line['business_id']
        if business_id in city_lookup:
            city = city_lookup[business_id]
            if funny + useful + cool > 0:
                if city == 'Las Vegas':
                    phoenix_count += 1
                    data[review_id] = {'text': text, 'city': 1, 'date': date, 'year': year, 'labels': {}}
                    data[review_id]['labels']['funny'] = {0: useful + cool, 1: funny}
                    data[review_id]['labels']['useful'] = {0: funny + cool, 1: useful}
                    data[review_id]['labels']['cool'] = {0: useful + funny, 1: cool}
                    data[review_id]['labels']['city'] = {0: 1, 1: 0}
                elif city == 'Toronto':
                    toronto_count += 1
                    data[review_id] = {'text': text, 'city': 0, 'date': date, 'year': year, 'labels': {}}
                    data[review_id]['labels']['funny'] = {0: useful + cool, 1: funny}
                    data[review_id]['labels']['useful'] = {0: funny + cool, 1: useful}
                    data[review_id]['labels']['cool'] = {0: useful + funny, 1: cool}
                    data[review_id]['labels']['city'] = {0: 0, 1: 1}

    print(toronto_count, phoenix_count)
    fh.makedirs(dirs.dir_data_raw(project_dir))

    fh.write_to_json(data, os.path.join(dirs.dir_data_raw(project_dir), 'all.json'))


if __name__ == '__main__':
    main()
