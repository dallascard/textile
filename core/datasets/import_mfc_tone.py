import os
import glob
import codecs
from optparse import OptionParser
from collections import defaultdict

from ..util import dirs
from ..util import file_handling as fh

def main():
    usage = "%prog project_name path/to/mfc_output.json output_prefix raw_data_dir metadata.json"
    parser = OptionParser(usage=usage)
    parser.add_option('-y', dest='year', default=2004,
                      help='Year at which to divide data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    data_file = args[1]
    output_prefix = args[2]
    raw_data_dir = args[3]
    metadata_file = args[4]

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

    keys = list(data.keys())
    for k in keys:
        text = data[k]['text']
        paragraphs = text.split('\n\n')
        text = '\n'.join(paragraphs[2:])
        tone_annotations = data[k]['annotations']['tone']
        year = int(data[k]['year'])
        month = int(data[k]['month'])
        source = data[k]['source']
        source = get_source(source)
        section = data[k]['section']
        csi = data[k]['csi']
        #framing_annotations = data[k]['annotations']['framing']
        article_tones = defaultdict(int)
        # process tone annotations
        for annotator, annotation_list in tone_annotations.items():
            for a in annotation_list:
                tone = TONE_CODES[int(a['code'])]
                if tone != 'Neutral':
                    if tone == 'Pro':
                        article_tones[1] += 1
                    else:
                        article_tones[0] += 1
        if len(article_tones) > 0 and year >= 1990:
            if year < threshold:
                year_group = 'pre_' + str(threshold)
            else:
                year_group = 'gte_' + str(threshold)
            year_group_sizes[year_group] += 1
            sources.add(source)
            sections.add(section)
            csis.add(csi)

            # only keep unanimous annotations
            #if len(article_tones) == 1:
            #    output[k] = {'text': text, 'label': int(list(article_tones.keys())[0]), 'year': int(year), 'year_group': year_group, 'month': month, 'source': source, 'csi': csi}

            # keep all annotations
            output[k] = {'text': text, 'label': article_tones, 'year': int(year), 'year_group': year_group, 'month': month, 'source': source}

    #print("Sources")
    sources = list(sources)
    sources.sort()
    #for s in sources:
    #    print(s)

    print(year_group_sizes)
    print(len(output))

    print("Loading non-annotated files")
    metadata = fh.read_json(metadata_file)

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

        if year < threshold:
            year_group = 'pre_' + str(threshold)
        else:
            year_group = 'gte_' + str(threshold)
        month = int(metadata[key]['month'])
        source = SOURCES[metadata[key]['source']]

        if key not in output:
            output[key] = {'text': text, 'label': {}, 'year': int(year), 'year_group': year_group, 'month': month, 'source': source}
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

