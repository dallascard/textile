import os
from optparse import OptionParser


from core.preprocessing import preprocess_labels
from core.preprocessing import preprocess_words
from core.preprocessing import preprocess_word_vectors


def main():
    usage = "%prog project subset word2vec_file"
    parser = OptionParser(usage=usage)
    #parser.add_option('--ref', dest='ref', default='unigrams',
    #                  help='Reference feature definition: default=%default')
    #parser.add_option('-s', dest='size', default=300,
    #                  help='Size of word vectors: default=%default')

    (options, args) = parser.parse_args()
    project = args[0]
    subset = args[1]
    word2vec_file = args[2]

    #subset = 'pro_tone'
    #base_project = os.path.join('projects', 'mfc')
    #subprojects = ['climate', 'guncontrol', 'immigration', 'samesex', 'smoking']

    preprocess_labels.preprocess_labels(project, subset, label_name='label', metadata_fields=['year_group'])
    preprocess_words.preprocess_words(project, subset, lower=True)
    preprocess_words.preprocess_words(project, subset, ngrams=1, lower=False, suffix='_default')
    preprocess_word_vectors.preprocess_word_vectors(project, subset, word2vec_file, ref='unigrams_default')

if __name__ == '__main__':
    main()
