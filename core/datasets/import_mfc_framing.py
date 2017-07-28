import os
from optparse import OptionParser
from collections import defaultdict

import numpy as np

from ..util import dirs
from ..util import file_handling as fh
from ..datasets.import_mfc_tone import get_source


CODES = {
    "1": "Economic",
    "2": "Capacity",
    "3": "Morality",
    "4": "Fairness",
    "5": "Legality",
    "6": "Policy",
    "7": "Crime",
    "8": "Security",
    "9": "Health",
    "10": "Quality",
    "11": "Cultural",
    "12": "Public",
    "13": "Political",
    "14": "External",
    "15": "Other",
}


def main():
    usage = "%prog project_name path/to/mfc_output.json output_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('-y', dest='n_years', default=5,
                      help='Number of years to group together: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    data_file = args[1]
    output_prefix = args[2]

    n_years = int(options.n_years)

    convert_mfc(project, data_file, output_prefix, n_years)


def convert_mfc(project, data_file, output_prefix, n_years):
    fh.makedirs(dirs.dir_data_raw(project))

    data = fh.read_json(data_file)
    output = {}
    sources = set()
    sections = set()
    csis = set()

    keys = list(data.keys())
    for k in keys:
        text = data[k]['text']
        paragraphs = text.split('\n\n')
        text = '\n'.join(paragraphs[2:])
        framing_annotations = data[k]['annotations']['framing']
        year = int(data[k]['year'])
        month = int(data[k]['month'])
        source = data[k]['source']
        source = get_source(source)
        section = data[k]['section']
        csi = data[k]['csi']
        n_annotations = 1
        annotation_counts = {}
        for i in range(1, 16):
            annotation_counts[i] = defaultdict(int)
        # process tone annotations
        for annotator, annotation_list in framing_annotations.items():
            frames = np.zeros(16)
            # look for presence of each frame
            for a in annotation_list:
                frame = int(a['code'])
                frames[frame] = 1
                n_annotations += 1
            # note the presence of legality annotations (0 or 1)
            for i in range(1, 16):
                annotation_counts[i][frames[i]] += 1

        if n_annotations > 0 and year >= 1990:
            year_lower = int(year / n_years) * n_years
            year_upper = year_lower + n_years - 1
            year_group = str(year_lower) + '-' + str(year_upper)
            sources.add(source)
            sections.add(section)
            csis.add(csi)

            output[k] = {'text': text, 'year': int(year), 'year_group': year_group, 'month': month, 'source': source}
            for i in range(1, 16):
                output[k][CODES[str(i)]] = annotation_counts[i]

    print("Sources")
    sources = list(sources)
    sources.sort()
    for s in sources:
        print(s)

    print("CSIs")
    csis = list(csis)
    csis.sort()
    for s in csis:
        print(s)

    print("Saving %d articles" % len(output))
    output_file = os.path.join(dirs.dir_data_raw(project), output_prefix + '.json')
    fh.write_to_json(output, output_filename=output_file)


if __name__ == '__main__':
    main()

