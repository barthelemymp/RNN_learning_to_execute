
import numpy as np
from tokenization import tokenize
from tokenization import build_vocabulary_token
from tokenization import vectorize_corpus
from tokenization import to_sentence
import csv
import torch
from numberseqseq import EncoderRNN
from numberseqseq import AttnDecoderRNN
from numberseqseq import model

# Load the useful arrays
fr_train=np.load('data_npy/fr_train.npy')
num_train=np.load('data_npy/num_train.npy')
rev_shared_vocab=np.load('data_npy/rev_shared_vocab.npy')
fr_val=np.load('data_npy/fr_val.npy')
num_val=np.load('data_npy/num_val.npy')
fr_test=np.load('data_npy/fr_test.npy')
num_test=np.load('data_npy/num_test.npy')

# Create the shared dictionary
tokenized_fr_train = [tokenize(s, word_level=True) for s in fr_train]
tokenized_num_train = [tokenize(s, word_level=False) for s in num_train]
shared_vocab, rev_shared_vocab = build_vocabulary_token(tokenized_fr_train+tokenized_num_train)


# Create the training, evaluating and testing sets
X_train, Y_train = vectorize_corpus(fr_train, num_train, shared_vocab,word_level_target=False)
X_val, Y_val = vectorize_corpus(fr_val, num_val, shared_vocab,word_level_target=False)
X_test, Y_test = vectorize_corpus(fr_test, num_test, shared_vocab,word_level_target=False)

pairs = [(torch.tensor(X_val[i], dtype=torch.long).view(-1, 1)
,torch.tensor(Y_val[i], dtype=torch.long).view(-1, 1)) for i in range(num_val.shape[0])]



