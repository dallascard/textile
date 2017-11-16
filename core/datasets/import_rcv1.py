import os
import glob
import zipfile
from xml.etree import ElementTree as ET
from optparse import OptionParser

from ..util import dirs
from ..util import file_handling as fh


def main():
    usage = "%prog project_name zipfile_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    project = args[0]
    input_dir = args[1]
    data = process_articles(input_dir)
    output_dir = dirs.dir_data_raw(project)
    fh.makedirs(output_dir)
    output_filename = os.path.join(output_dir, 'all.json')
    fh.write_to_json(data, output_filename)


def process_articles(input_dir):

    data = {}

    files = glob.glob(os.path.join(input_dir, '199*.zip'))
    files.sort()
    for file in files:
        with zipfile.ZipFile(file, 'r') as f:
            names = f.namelist()
            print('%s (%d)' % (file, len(names)))
            n_missing_codes = 0
            for name in names:
                title = ''
                headline = ''
                text = ''
                codes = []
                try:
                    xml = f.read(name)
                    root = ET.fromstring(xml)
                    attributes = root.attrib
                    id = 'rcv' + str(attributes['itemid'])
                    date = attributes['date']
                    year = int(date.split('-')[0])
                    for child in root:
                        if child.tag == 'title':
                            if child.text is not None:
                                title = child.text
                        elif child.tag == 'headline':
                            if child.text is not None:
                                headline = child.text
                        elif child.tag == 'text':
                            for paragraph in child:
                                if child.text is not None:
                                    if text != '':
                                        text += '\n\n'
                                    text += paragraph.text
                        elif child.tag == 'metadata':
                            for subchild in child:
                                if subchild.tag == 'codes' and subchild.attrib['class'] == 'bip:topics:1.0':
                                    for code in subchild:
                                        codes.append(code.attrib['code'])
                    if text == '':
                        print("Text is empty")
                    if title == '':
                        print("Title is empty")
                    if headline == '':
                        print("Headline is empty")
                    if len(codes) == 0:
                        n_missing_codes += 1
                    data[id] = {'text': title + '\n\n' + headline + '\n\n' + text, 'date': date, 'year': year, 'labels': {}}
                    for code in codes:
                        data[id]['labels'][code] = 1
                except Exception as e:
                    print(name, e)
            print("missing codes: %d" % n_missing_codes)

    return data


if __name__ == '__main__':
    main()
