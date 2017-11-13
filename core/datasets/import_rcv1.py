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
    output_filename = os.path.join(dirs.dir_data_raw(project), 'all.json')
    fh.write_to_json(data, output_filename)


def process_articles(input_dir):

    data = {}

    files = glob.glob(os.path.join(input_dir, '199*.zip'))
    files.sort()
    for file in files:
        print(file)
        with zipfile.ZipFile(file, 'r') as f:
            names = f.namelist()
            for name in names:
                title = ''
                headline = ''
                text = ''
                codes = []
                xml = f.read(name)
                root = ET.fromstring(xml)
                attributes = root.attrib
                id = attributes['itemid']
                date = attributes['date']
                year = int(date.split('-')[0])
                for child in root:
                    if child.tag == 'title':
                        title = child.text
                    elif child.tag == 'headline':
                        headline = child.text
                    elif child.tag == 'text':
                        for paragraph in child:
                            if text != '':
                                text += '\n\n'
                            text = paragraph.text
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
                    print("Codes is empty")
                data[id] = {'text': title + '\n\n' + headline + '\n\n' + text, 'date': date, 'year': year, 'codes': codes}

    return data


if __name__ == '__main__':
    main()
