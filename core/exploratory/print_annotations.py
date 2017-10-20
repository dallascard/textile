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
    usage = "%prog path/to/documents.json index target_frame"
    parser = OptionParser(usage=usage)

    (options, args) = parser.parse_args()

    data_file = args[0]
    index = int(args[1])
    target_frame = args[2]

    data = fh.read_json(data_file)

    keys = list(data.keys())
    keys.sort()

    phrases = []

    while len(phrases) == 0:
        k = keys[index]

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
                    phrases.append(annotator + ':' + text[start:end])

        if len(phrases) > 0:
            print(index)
            print('\n'.join(phrases))

        index += 1

    print()


if __name__ == '__main__':
    main()

