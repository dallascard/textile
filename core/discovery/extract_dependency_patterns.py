import os
import re
import sys
from collections import Counter
from collections import defaultdict
from optparse import OptionParser

import numpy as np
import pandas as pd
from sklearn import preprocessing
from spacy.en import English

from ..util import file_handling as fh
from ..preprocessing import features
from ..util import dirs


def main():
    usage = "%prog project_dir subset"
    parser = OptionParser(usage=usage)
    parser.add_option('--lemmatize', action="store_true", dest="lemmatize", default=False,
                      help='Use lemmas instead of words: default=%default')
    parser.add_option('--tags', action="store_true", dest="tags", default=False,
                      help='Use rich part of speech tags: default=%default')
    parser.add_option('--lower', action="store_true", dest="lower", default=False,
                      help='Lower case words: default=%default')
    #parser.add_option('--reference_subset', dest='reference', default=None,
    #                  help='Reference subset (i.e. train) for ensuring full : default=%default')

    (options, args) = parser.parse_args()

    project_dir = args[0]
    subset = args[1]
    datafile = os.path.join(dirs.dir_data_raw(project_dir), subset + '.json')
    lemmatize = options.lemmatize
    tags = options.tags
    lower = options.lower
    extract_patterns(datafile, lemmatize, tags, lower)


def extract_patterns(datafile, lemmatize=False, tags=False, lower=False):

    print("Reading data")
    data = fh.read_json(datafile)

    print("Loading spacy")
    parser = English()
    #tagger = phrasemachine.get_stdeng_nltk_tagger()

    labels = []
    unigrams = defaultdict(Counter)
    bigrams = defaultdict(Counter)
    #trigrams = defaultdict(Counter)

    print("Parsing texts")
    keys = list(data.keys())
    keys.sort()

    for k_i, key in enumerate(keys[:1]):
        if k_i % 100 == 0 and k_i > 0:
            print(k_i)

        item = data[key]
        text = item['text']
        # replace underscores with dashes to avoid confusion
        text = re.sub('_', '-', text)

        # parse the text with spaCy
        parse = parser(text)
        sents = list(parse.sents)

        max_len = 5
        fragments = []
        for sent in sents[2:3]:
            tokens = [t for t in sent if t.pos_ != 'SPACE' and t.pos_ != 'PUNCT']
            fragments_to_expand = [Fragment(node=t) for t in tokens]
            while len(fragments_to_expand) > 0:
                fragment = fragments_to_expand.pop()
                if len(fragment) < max_len:
                    fragments_to_expand.extend(fragment.get_sub_fragments())
                print(fragment)
                fragments.append(fragment)


        """

        for token in parse:
            word = get_word(token, lemmatize=lemmatize, lower=lower)
            pos = get_pos_tag(token, tags=tags)
            dep = token.dep_
            index = token.i
            parent = token.head
            children = list(token.children)
            if len(children) == 0:
                parent_word = get_word(parent, lemmatize=lemmatize, lower=lower)
                parent_pos = get_pos_tag(parent, tags=tags)
                parent_index = parent.i
                dep_pattern = '_'.join([parent_pos, dep, pos, str(parent_index - index)])

                #children_tokens = token.children
                #ent_type = token.ent_type_
                #prefix = token.prefix_
                #suffix = token.suffix_

                if pos not in ['SPACE', 'PUNCT'] and parent_pos not in ['SPACE', 'PUNCT']:
                    unigrams[pos].update([word])
                    if index != parent_index:
                        if index < parent_index:
                            bigrams[dep_pattern].update(['_'.join([word, parent_word])])
                            #bigrams[dep_pattern].update(['_'.join([word, parent_pos])])
                            #bigrams[dep_pattern].update(['_'.join([pos, parent_word])])
                        else:
                            bigrams[dep_pattern].update(['_'.join([parent_word, word])])
                            #bigrams[dep_pattern].update(['_'.join([parent_pos, word])])
                            #bigrams[dep_pattern].update(['_'.join([parent_word, pos])])

        """

    """
    output_dir = dirs.dir_patterns(project_dir, subset)
    fh.makedirs(output_dir)
    fh.write_to_json(unigrams, os.path.join(output_dir, 'unigrams.json'))

    output_dir = dirs.dir_patterns(project_dir, subset)
    fh.makedirs(output_dir)
    fh.write_to_json(bigrams, os.path.join(output_dir, 'bigrams.json'))
    """


def get_word(token, lemmatize=False, lower=False):
    #  get word and remove whitespace
    if lemmatize:
        word = re.sub('\s', '', token.lemma_)
    else:
        word = re.sub('\s', '', token.orth_)
    if lower:
        word = word.lower()
    return word


def get_pos_tag(token, tags=False):
    if tags:
        return token.tag_
    else:
        return token.pos_


def extract_fragments_from_tree(tokens, max_size=4):
    fragments = []
    root = get_tree_root(tokens)


def get_tree_root(tokens):
    t = tokens[0]
    while t.dep_ != 'ROOT':
        t = t.head
    return t


def get_child_sets(token, max_children):
    sets = []


def get_sibling_sets(tokens, index, size):
    sets = []


class Fragment:

    def __init__(self, node=None, nodes=None):
        self.nodes = []
        self.children = []
        if node is not None:
            self.add_node(node)
        if nodes is not None:
            self.nodes = nodes[:]
            #self.children = fragment.children[:]

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        indices = [n.i for n in self.nodes]
        indices.sort()
        min_index = indices[0]
        max_index = indices[-1]
        n_terms = max_index - min_index + 1
        parts = ['_'] * n_terms
        for n in self.nodes:
            parts[n.i - min_index] = n.orth_
        return '[' + ' '.join(parts) + ']'

    def add_node(self, node):
        self.nodes.append(node)
        children = list(node.children)
        children = [c for c in children if c.pos_ != 'SPACE' and c.pos_ != 'PUNCT']
        self.children.extend(children)

    def add_children(self, children):
        self.children.extend(children[:])

    def get_sub_fragments(self):
        fragments = []
        children = self.children[:]
        while len(children) > 0:
            c = children.pop()
            fragment = Fragment(nodes=self.nodes)
            fragment.add_node(c)
            fragment.add_children(children)
            fragments.append(fragment)
        return fragments


if __name__ == '__main__':
    main()
