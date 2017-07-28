import os
from optparse import OptionParser
from collections import defaultdict

from ..util import dirs
from ..util import file_handling as fh

TONE_CODES = {17: 'Pro', 18: 'Neutral', 19: 'Anti'}


SOURCES = {
    'atlanta journal and constitution': 'Atlanta_Journal_and_Constitution',
    'atlanta journal-constitution': 'Atlanta_Journal_and_Constitution',
    'daily news (new york)': 'NY_Daily_News',
    'denver post': 'Denver_Post',
    'denver post the denver post': 'Denver_Post',
    'herald-sun (durham, n.c.)': 'Herald-Sun',
    'herald-sun (durham, nc)': 'Herald-Sun',
    'raleigh extra (durham, nc)': 'Herald-Sun',
    'new york times': 'NY_Times',
    'palm beach post (florida)': 'Palm_Beach_Post',
    'philadelphia inquirer': 'Philadelphia_Inquirer',
    'saint paul pioneer press (minnesota)': 'St._Paul_Pioneer_Press',
    'san jose mercury news (california)': 'San_Jose_Mercury_News',
    'st. louis post-dispatch (missouri)': 'St._Louis_Post-Dispatch',
    'st. paul pioneer press (minnesota)': 'St._Paul_Pioneer_Press',
    'st. petersburg times (florida)': 'Tampa_Bay_Times',  # renamed
    'tampa bay times': 'Tampa_Bay_Times',
    'usa today': 'USA_Today',
    'washington post': 'Washington_Post',
    'washingtonpost.com': 'Washington_Post'
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
            year_lower = int(year / n_years) * n_years
            year_upper = year_lower + n_years - 1
            year_group = str(year_lower) + '-' + str(year_upper)
            sources.add(source)
            sections.add(section)
            csis.add(csi)

            # only keep unanimous annotations
            #if len(article_tones) == 1:
            #    output[k] = {'text': text, 'label': int(list(article_tones.keys())[0]), 'year': int(year), 'year_group': year_group, 'month': month, 'source': source, 'csi': csi}

            # keep all annotations
            output[k] = {'text': text, 'label': article_tones, 'year': int(year), 'year_group': year_group, 'month': month, 'source': source}

    print("Sources")
    sources = list(sources)
    sources.sort()
    for s in sources:
        print(s)

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

