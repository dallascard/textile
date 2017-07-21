import os
from optparse import OptionParser
from collections import defaultdict

from ..util import dirs
from ..util import file_handling as fh

TONE_CODES = {17: 'Pro', 18: 'Neutral', 19: 'Anti'}

def main():
    usage = "%prog project_name path/to/mfc_output.json output_prefix"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    project = args[0]
    data_file = args[1]
    output_prefix = args[2]

    convert_mfc(project, data_file, output_prefix)


def convert_mfc(project, data_file, output_prefix):
    fh.makedirs(dirs.dir_data_raw(project))

    data = fh.read_json(data_file)
    output = {}

    keys = list(data.keys())
    for k in keys:
        text = data[k]['text']
        paragraphs = text.split('\n\n')
        text = '\n'.join(paragraphs[2:])
        tone_annotations = data[k]['annotations']['tone']
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
        if len(article_tones) > 0:
            year = int(data[k]['year'])
            year_lower = int(year / 3) * 3
            year_upper = year_lower + 2
            year_group = str(year_lower) + '-' + str(year_upper)

            # only keep unanimous annotations
            if len(article_tones) == 1:
                output[k] = {'text': text, 'label': int(list(article_tones.keys())[0]), 'year': int(year), 'year_group': year_group}

            # keep all annotations
            #output[k] = {'text': text, 'label': article_tones, 'year': int(year), 'year_group': year_group}

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

