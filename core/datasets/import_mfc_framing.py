import os
import glob
import codecs
from optparse import OptionParser
from collections import defaultdict

import numpy as np

from ..util import dirs
from ..util import file_handling as fh

FRAMES =   ["Economic",
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
    usage = "%prog project_name path/to/documents.json raw_data_dir metadata.json"
    parser = OptionParser(usage=usage)
    parser.add_option('-y', dest='year', default=2004,
                      help='Year at which to divide data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    data_file = args[1]
    output_prefix = 'framing'
    raw_data_dir = args[2]
    metadata_file = args[3]

    threshold = int(options.year)

    convert_mfc(project, data_file, output_prefix, threshold, raw_data_dir, metadata_file)


def convert_mfc(project, data_file, output_prefix, threshold, raw_data_dir, metadata_file):
    fh.makedirs(dirs.dir_data_raw(project))

    data = fh.read_json(data_file)
    output = {}
    sources = set()
    sections = set()
    csis = set()
    year_group_sizes = defaultdict(int)

    metadata = fh.read_json(metadata_file)

    total_frames = np.zeros(n_frames)

    keys = list(data.keys())
    keys.sort()

    frame_counts = np.zeros(len(FRAMES))

    for k in keys:
        text = data[k]['text']
        paragraphs = text.split('\n\n')
        text = ' '.join(paragraphs[2:])
        framing_annotations = data[k]['annotations']['framing']
        #year = int(data[k]['year'])
        #month = int(data[k]['month'])
        #source = data[k]['source']
        #source = get_source(source)
        #section = data[k]['section']
        #csi = data[k]['csi']
        #framing_annotations = data[k]['annotations']['framing']
        article_frames = np.zeros([n_frames, 2])
        # process framing annotations
        for annotator, annotation_list in framing_annotations.items():
            annotator_frames = np.zeros(n_frames)
            # look for presence of each frame
            for a in annotation_list:
                frame = int(a['code']) - 1
                annotator_frames[frame] = 1

            # note the presence or absence of each frame for this annotator
            article_frames[:, 1] += annotator_frames
            article_frames[:, 0] += 1 - annotator_frames

        year = int(metadata[k]['year'])
        month = int(metadata[k]['month'])
        source = SOURCES[metadata[k]['source']]

        if np.sum(article_frames) > 0 and year >= 1990:
            if year < threshold:
                year_group = 'pre_' + str(threshold)
            else:
                year_group = 'gte_' + str(threshold)
            year_group_sizes[year_group] += 1
            sources.add(source)
            total_frames += article_frames[:, 1]

            # only keep unanimous annotations
            #if len(article_tones) == 1:
            #    output[k] = {'text': text, 'label': int(list(article_tones.keys())[0]), 'year': int(year), 'year_group': year_group, 'month': month, 'source': source, 'csi': csi}

            # keep all annotations
            output[k] = {'text': text, 'year': int(year), 'year_group': year_group, 'month': month, 'source': source, 'labels': {}}

            for frame_i, frame in enumerate(FRAMES):
                output[k]['labels'][frame] = {0: article_frames[frame_i, 0], 1: article_frames[frame_i, 1]}
                frame_counts[frame_i] += article_frames[frame_i, 1]

    for frame_i, frame in enumerate(FRAMES):
        print(frame, frame_counts[frame_i])

    print(year_group_sizes)
    print(len(output))

    total_frames = total_frames / float(len(output))
    for f_i, f in enumerate(FRAMES):
        print(f, total_frames[f_i])

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

        if key not in output and year >= 1990:
            if year < threshold:
                year_group = 'pre_' + str(threshold)
            else:
                year_group = 'gte_' + str(threshold)
            month = int(metadata[key]['month'])
            source = SOURCES[metadata[key]['source']]

            output[key] = {'text': text, 'year': int(year), 'year_group': year_group, 'month': month, 'source': source, 'labels': {}}
            for frame_i, frame in enumerate(FRAMES):
                output[key]['labels'][frame] = {}

            year_group_sizes[year_group] += 1

    print(year_group_sizes)
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

