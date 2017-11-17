import os
import glob
import codecs
from optparse import OptionParser
from collections import defaultdict

from ..util import dirs
from ..util import file_handling as fh

TONE_CODES = {17: 'Pro', 18: 'Neutral', 19: 'Anti'}

# same as before, but designed to work directly from the output of scripts in compuframe-coding/tools/


SOURCES = {
    'atlanta journal and constitution': 'Atlanta_Journal_and_Constitution',
    'atlanta journal-constitution': 'Atlanta_Journal_and_Constitution',
    'daily news (new york)': 'NY_Daily_News',
    'daily news': 'NY_Daily_News',
    'denver post': 'Denver_Post',
    'denver post the denver post': 'Denver_Post',
    'herald-sun (durham, n.c.)': 'Herald-Sun',
    'herald-sun (durham, nc)': 'Herald-Sun',
    'herald-sun': 'Herald-Sun',
    'chapel hill herald': 'Herald-Sun',
    'raleigh extra (durham, nc)': 'Herald-Sun',
    'raleigh extra': 'Herald-Sun',
    'new york times': 'NY_Times',
    'palm beach post (florida)': 'Palm_Beach_Post',
    'palm beach post': 'Palm_Beach_Post',
    'philadelphia inquirer': 'Philadelphia_Inquirer',
    'saint paul pioneer press (minnesota)': 'St._Paul_Pioneer_Press',
    'saint paul pioneer press': 'St._Paul_Pioneer_Press',
    'san jose mercury news (california)': 'San_Jose_Mercury_News',
    'san jose mercury news': 'San_Jose_Mercury_News',
    'st. louis post-dispatch (missouri)': 'St._Louis_Post-Dispatch',
    'st. louis post-dispatch': 'St._Louis_Post-Dispatch',
    'st. paul pioneer press (minnesota)': 'St._Paul_Pioneer_Press',
    'st. petersburg times (florida)': 'Tampa_Bay_Times',  # renamed
    'st. petersburg times': 'Tampa_Bay_Times',  # renamed
    'tampa bay times': 'Tampa_Bay_Times',
    'usa today': 'USA_Today',
    'washington post': 'Washington_Post',
    'washingtonpost.com': 'Washington_Post'
}

def main():
    usage = "%prog project_name raw_data_dir metadata.json"
    parser = OptionParser(usage=usage)
    #parser.add_option('-y', dest='year', default=2004,
    #                  help='Year at which to divide data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    output_prefix = 'sources'
    raw_data_dir = args[1]
    metadata_file = args[2]

    convert_mfc(project, output_prefix, raw_data_dir, metadata_file)


def convert_mfc(project, output_prefix, raw_data_dir, metadata_file):
    fh.makedirs(dirs.dir_data_raw(project))

    output = {}
    metadata = fh.read_json(metadata_file)

    print("Loading non-annotated files")
    raw_files = glob.glob(os.path.join(raw_data_dir, '*.txt'))
    for f_i, f in enumerate(raw_files):
        if f_i % 1000 == 0 and f_i > 0:
            print(f_i)
        filename = os.path.split(f)[1]
        key = filename.split('_short.txt')[0]
        with codecs.open(f, 'r') as input_file:
            text = input_file.read()
        paragraphs = text.split('\n\n')
        text = '\n'.join(paragraphs[2:])
        year = int(metadata[key]['year'])

        month = int(metadata[key]['month'])
        source = SOURCES[metadata[key]['source']]

        output[key] = {'text': text, 'labels': {}, 'year': int(year), 'month': month, 'source': source}
        output[key]['labels'][source] = 1

    print(len(output))

    print("Saving %d articles" % len(output))
    output_file = os.path.join(dirs.dir_data_raw(project), output_prefix + '.json')
    fh.write_to_json(output, output_filename=output_file)


def get_source(source):
    if source.startswith('new york times blogs'):
        source = 'NY_Times_blogs'
    elif source.startswith('washington post blogs'):
        source = 'Washington_Post_blogs'
    else:
        source = SOURCES[source]
    return source



if __name__ == '__main__':
    main()

