# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 19:25:09 2019

@author: barthelemy
"""

import numpy as np
from tokenization import tokenize
from tokenization import build_vocabulary_token
from tokenization import vectorize_corpus
import csv
import time

fr_train=np.load('data_npy/fr_train.npy')
num_train=np.load('data_npy/num_train.npy')
rev_shared_vocab=np.load('data_npy/rev_shared_vocab.npy')
fr_val=np.load('data_npy/fr_val.npy')
num_val=np.load('data_npy/num_val.npy')
fr_test=np.load('data_npy/fr_test.npy')
num_test=np.load('data_npy/num_test.npy')

print(fr_train[:10])
print(num_train[:10])

# Create the shared dictionary
tokenized_fr_train = [tokenize(s, word_level=True) for s in fr_train]
tokenized_num_train = [tokenize(s, word_level=False) for s in num_train]
shared_vocab, rev_shared_vocab = build_vocabulary_token(tokenized_fr_train+tokenized_num_train)

# Create the training, evaluating and testing sets
X_train, Y_train = vectorize_corpus(fr_train, num_train, shared_vocab,word_level_target=False)
X_val, Y_val = vectorize_corpus(fr_val, num_val, shared_vocab,word_level_target=False)
X_test, Y_test = vectorize_corpus(fr_test, num_test, shared_vocab,word_level_target=False)


print(X_train[:10])
print(Y_train[:10])



pairs = [[fr_train[i],num_train[i]] for i in range(num_train.shape[0])]
print('bla')
print(pairs[:10])




#from __future__ import unicode_literals, print_function, division
#from io import open
#import unicodedata
#import string
#import re
#import random
#
#import torch
#import torch.nn as nn
#from torch import optim
#import torch.nn.functional as F
#
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#######################################################################
## Loading data files
## ==================
##
## The data for this project is a set of many thousands of English to
## French translation pairs.
##
## `This question on Open Data Stack
## Exchange <https://opendata.stackexchange.com/questions/3888/dataset-of-sentences-translated-into-many-languages>`__
## pointed me to the open translation site https://tatoeba.org/ which has
## downloads available at https://tatoeba.org/eng/downloads - and better
## yet, someone did the extra work of splitting language pairs into
## individual text files here: https://www.manythings.org/anki/
##
## The English to French pairs are too big to include in the repo, so
## download to ``data/eng-fra.txt`` before continuing. The file is a tab
## separated list of translation pairs:
##
## ::
##
##     I am cold.    J'ai froid.
##
## .. Note::
##    Download the data from
##    `here <https://download.pytorch.org/tutorial/data.zip>`_
##    and extract it to the current directory.
#
#######################################################################
## Similar to the character encoding used in the character-level RNN
## tutorials, we will be representing each word in a language as a one-hot
## vector, or giant vector of zeros except for a single one (at the index
## of the word). Compared to the dozens of characters that might exist in a
## language, there are many many more words, so the encoding vector is much
## larger. We will however cheat a bit and trim the data to only use a few
## thousand words per language.
##
## .. figure:: /_static/img/seq-seq-images/word-encoding.png
##    :alt:
##
##
#
#
#######################################################################
## We'll need a unique index per word to use as the inputs and targets of
## the networks later. To keep track of all this we will use a helper class
## called ``Lang`` which has word → index (``word2index``) and index → word
## (``index2word``) dictionaries, as well as a count of each word
## ``word2count`` to use to later replace rare words.
##
#
#SOS_token = 0
#EOS_token = 1
#
#
#class Lang:
#    def __init__(self, name):
#        self.name = name
#        self.word2index = {}
#        self.word2count = {}
#        self.index2word = {0: "SOS", 1: "EOS"}
#        self.n_words = 2  # Count SOS and EOS
#
#    def addSentence(self, sentence):
#        for word in sentence.split(' '):
#            self.addWord(word)
#
#    def addWord(self, word):
#        if word not in self.word2index:
#            self.word2index[word] = self.n_words
#            self.word2count[word] = 1
#            self.index2word[self.n_words] = word
#            self.n_words += 1
#        else:
#            self.word2count[word] += 1
#
#
#######################################################################
## The files are all in Unicode, to simplify we will turn Unicode
## characters to ASCII, make everything lowercase, and trim most
## punctuation.
##
#
## Turn a Unicode string to plain ASCII, thanks to
## https://stackoverflow.com/a/518232/2809427
#def unicodeToAscii(s):
#    return ''.join(
#        c for c in unicodedata.normalize('NFD', s)
#        if unicodedata.category(c) != 'Mn'
#    )
#
## Lowercase, trim, and remove non-letter characters
#
#
#def normalizeString(s):
#    s = unicodeToAscii(s.lower().strip())
#    s = re.sub(r"([.!?])", r" \1", s)
#    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#    return s
#
#
#######################################################################
## To read the data file we will split the file into lines, and then split
## lines into pairs. The files are all English → Other Language, so if we
## want to translate from Other Language → English I added the ``reverse``
## flag to reverse the pairs.
##
#
#def readLangs(lang1, lang2, reverse=False):
#    print("Reading lines...")
#
#    # Read the file and split into lines
#    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
#        read().strip().split('\n')
#
#    # Split every line into pairs and normalize
#    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
#
#    # Reverse pairs, make Lang instances
#    if reverse:
#        pairs = [list(reversed(p)) for p in pairs]
#        input_lang = Lang(lang2)
#        output_lang = Lang(lang1)
#    else:
#        input_lang = Lang(lang1)
#        output_lang = Lang(lang2)
#
#    return input_lang, output_lang, pairs
#
#
#######################################################################
## Since there are a *lot* of example sentences and we want to train
## something quickly, we'll trim the data set to only relatively short and
## simple sentences. Here the maximum length is 10 words (that includes
## ending punctuation) and we're filtering to sentences that translate to
## the form "I am" or "He is" etc. (accounting for apostrophes replaced
## earlier).
##
#
#MAX_LENGTH = 10
#
#eng_prefixes = (
#    "i am ", "i m ",
#    "he is", "he s ",
#    "she is", "she s ",
#    "you are", "you re ",
#    "we are", "we re ",
#    "they are", "they re "
#)
#
#
#def filterPair(p):
#    return len(p[0].split(' ')) < MAX_LENGTH and \
#        len(p[1].split(' ')) < MAX_LENGTH and \
#        p[1].startswith(eng_prefixes)
#
#
#def filterPairs(pairs):
#    return [pair for pair in pairs if filterPair(pair)]
#
#
#######################################################################
## The full process for preparing the data is:
##
## -  Read text file and split into lines, split lines into pairs
## -  Normalize text, filter by length and content
## -  Make word lists from sentences in pairs
##
#
#def prepareData(lang1, lang2, reverse=False):
#    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
#    print("Read %s sentence pairs" % len(pairs))
#    pairs = filterPairs(pairs)
#    print("Trimmed to %s sentence pairs" % len(pairs))
#    print("Counting words...")
#    for pair in pairs:
#        input_lang.addSentence(pair[0])
#        output_lang.addSentence(pair[1])
#    print("Counted words:")
#    print(input_lang.name, input_lang.n_words)
#    print(output_lang.name, output_lang.n_words)
#    return input_lang, output_lang, pairs
#
#
#input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
#print(random.choice(pairs))
#
#
