import os
import glob
import codecs
from optparse import OptionParser
from collections import defaultdict

import numpy as np

from ..util import dirs
from ..util import file_handling as fh

FRAMES = ["Economic",
          "Capacity",
          "Morality",
          "Fairness",
          "Legality",
          "Policy",
          "Crime",
          "Security",
          "Health",
          "Quality",
          "Cultural",
          "Public",
          "Political",
          "External",
          "Other"]

CODES = {str(int(i+1)): f for i, f in enumerate(FRAMES)}

n_frames = len(FRAMES)

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
    usage = "%prog project_name path/to/documents.json metadata.json target_frame"
    parser = OptionParser(usage=usage)
    parser.add_option('-y', dest='year', default=2004,
                      help='Year at which to divide data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    data_file = args[1]
    metadata_file = args[2]
    target_frame = args[3]
    output_prefix = target_frame + '_annotations'

    threshold = int(options.year)

    convert_mfc(project, data_file, output_prefix, threshold, metadata_file, target_frame)


def convert_mfc(project, data_file, output_prefix, threshold, metadata_file, target_frame):
    fh.makedirs(dirs.dir_data_raw(project))

    data = fh.read_json(data_file)
    output = {}
    sources = set()
    year_group_sizes = defaultdict(int)

    metadata = fh.read_json(metadata_file)

    keys = list(data.keys())
    keys.sort()

    for k in keys:
        words = []
        text = data[k]['text']
        framing_annotations = data[k]['annotations']['framing']
        # extract all annotations, double counting for doubly-annotated
        for annotator, annotation_list in framing_annotations.items():
            # look for presence of each frame
            for a in annotation_list:
                frame = int(a['code']) - 1
                start = int(a['start'])
                end = int(a['end'])
                if FRAMES[frame] == target_frame:
                    words.extend(text[start:end].split())

        year = int(metadata[k]['year'])
        month = int(metadata[k]['month'])
        source = SOURCES[metadata[k]['source']]

        # only export those items with annotations
        if len(words) > 0 and year >= 1990:
            if year < threshold:
                year_group = 'pre_' + str(threshold)
            else:
                year_group = 'gte_' + str(threshold)
            year_group_sizes[year_group] += 1
            sources.add(source)

            # keep all annotations
            output[k] = {'text': ' '.join(words), 'year': int(year), 'year_group': year_group, 'month': month, 'source': source}

    print(year_group_sizes)
    print(len(output))

    print("Loading non-annotated files")

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

