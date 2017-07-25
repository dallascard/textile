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
    'herald-sun (durham, n.c.)': 'Herald-Sun',
    'herald-sun (durham, nc)': 'Herald-Sun',
    'raleigh extra (durham, nc)': 'Herald-Sun',
    'new york times': 'NY_Times',
    'new york times blogs (carpetbagger)': 'NY_Times_blogs',
    'new york times blogs (city room)': 'NY_Times_blogs',
    'new york times blogs (taking note)': 'NY_Times_blogs',
    'new york times blogs (the caucus)': 'NY_Times_blogs',
    'new york times blogs (the learning network)': 'NY_Times_blogs',
    'new york times blogs (the loyal opposition)': 'NY_Times_blogs',
    'new york times blogs (the lede)': 'NY_Times_blogs',
    'new york times blogs (lens)': 'NY_Times_blogs',
    'new york times blogs (iht rendezvous)': 'NY_Times_blogs',
    'new york times blogs (opinionator)': 'NY_Times_blogs',
    'new york times blogs (ross douthat)': 'NY_Times_blogs',
    'new york times blogs (india ink)': 'NY_Times_blogs',
    'new york times blogs (campaign stops)': 'NY_Times_blogs',
    'new york times blogs (prescriptions)': 'NY_Times_blogs',
    'new york times blogs (economix)': 'NY_Times_blogs',
    'new york times blogs (well)': 'NY_Times_blogs',
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
    'washingtonpost.com': 'Washington_Post',
    'washington post blogs election 2012': 'Washington_Post_blogs'
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
        source = SOURCES[data[k]['source']]
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
            if len(article_tones) == 1:
                output[k] = {'text': text, 'label': int(list(article_tones.keys())[0]), 'year': int(year), 'year_group': year_group, 'month': month, 'source': source, 'csi': csi}

            # keep all annotations
            #output[k] = {'text': text, 'label': article_tones, 'year': int(year), 'year_group': year_group}

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


CODES = {
    "0": "None",
    "1.0": "Economic",
    "1.1": "Economic headline",
    "1.2": "Economic primary",
    "2.0": "Capacity and Resources",
    "2.1": "Capacity and Resources headline",
    "2.2": "Capacity and Resources primany",
    "3.0": "Morality",
    "3.1": "Morality headline",
    "3.2": "Morality primary",
    "4.0": "Fairness and Equality",
    "4.1": "Fairness and Equality headline",
    "4.2": "Fairness and Equality primary",
    "5.0": "Legality, Constitutionality, Jurisdiction",
    "5.1": "Legality, Constitutionality, Jurisdiction headline",
    "5.2": "Legality, Constitutionality, Jurisdiction primary",
    "6.0": "Policy Prescription and Evaluation",
    "6.1": "Policy Prescription and Evaluation headline",
    "6.2": "Policy Presecription and Evaluation primary",
    "7.0": "Crime and Punishment",
    "7.1": "Crime and Punishment headline",
    "7.2": "Crime and Punishment primary",
    "8.0": "Security and Defense",
    "8.1": "Security and Defense headline",
    "8.2": "Security and Defense primary",
    "9.0": "Health and Safety",
    "9.1": "Health and Safety headline",
    "9.2": "Health and Safety primary",
    "10.0": "Quality of Life",
    "10.1": "Quality of life headline",
    "10.2": "Quality of Life primary",
    "11.0": "Cultural Identity",
    "11.1": "Cultural Identity headline",
    "11.2": "Cultural Identity primary",
    "12.0": "Public Sentiment",
    "12.1": "Public Sentiment headline",
    "12.2": "Public Sentiment primary",
    "13.0": "Political",
    "13.1": "Political primary headline",
    "13.2": "Political primary",
    "14.0": "External Regulation and Reputation",
    "14.1": "External regulation and reputation headline",
    "14.2": "External Regulation and Reputation primary",
    "15.0": "Other",
    "15.1": "Other headline",
    "15.2": "Other primary",
    "16.2": "Irrelevant",
    "17.35": "Implicit Pro",
    "17.4": "Explicit Pro",
    "18.3": "Neutral",
    "19.35": "Implicit Anti",
    "19.4": "Explicit Anti"
}

