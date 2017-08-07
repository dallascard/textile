from optparse import OptionParser

from collections import defaultdict, Counter

import numpy
from scipy import sparse

def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()



def do_byte_pair_encoding(text, vocab_size):
    n = len(text)

    vocab = list(set(n))
    vocab.sort()
    vocab_index = dict(zip(vocab, range(len(vocab))))

    # convert text to a list of indices
    text = [vocab_index[c] for c in text]

    counts = Counter()
    text_index = defaultdict(list)

    # build text index and initial count matrix
    for i in range(len(text)-1):
        pair = text[i:i+2]
        index = vocab_index[pair]
        if pair not in vocab_index:
            vocab_index[pair] = len(vocab)
            vocab.append(pair)
        text_index[index].append(i)
        counts.update([index])

    # find most common pair
    most_common = counts.most_common(n=1)[0][0]
    index = vocab_index[most_common]
    text_indices = text_index[index]



    # remove the count for the pair
    counts.pop(most_common)








if __name__ == '__main__':
    main()
