from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
#import matplotlib.pyplot as plt

import numpy as np
from utils import tokenize
from utils import build_vocabulary_token
from utils import vectorize_corpus
import csv
import time

from model import EncoderRNN
from model import AttnDecoderRNN
from model import model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


#############OTHER DATA PREP

#pairs = [[fr_train[i],num_train[i]] for i in range(num_train.shape[0])]

pairs = [(torch.tensor(X_train[i], dtype=torch.long, device=device).view(-1, 1)
,torch.tensor(Y_train[i], dtype=torch.long, device=device).view(-1, 1)) for i in range(num_train.shape[0])]

test_pairs=[(torch.tensor(X_test[i], dtype=torch.long, device=device).view(-1, 1)
,torch.tensor(Y_test[i], dtype=torch.long, device=device).view(-1, 1)) for i in range(num_test.shape[0])]

score_pairs = [(torch.tensor(X_val[i], dtype=torch.long, device=device).view(-1, 1)
,torch.tensor(Y_val[i], dtype=torch.long, device=device).view(-1, 1)) for i in range(num_test.shape[0])]


MAX_LENGTH = 20


GO_token = 1
EOS_token = 2

config ={
        'dropout': 0.2,
        'vocab_size': 40,
        'num_layers': 1,
        'embsize': 32,
        'dim_recurrent': 256,
        'batch_size':32,
        'hidden_size': 256,
        'max_length': 20
    }


encoder=EncoderRNN(input_size=config["vocab_size"], hidden_size=config["hidden_size"]).to(device)
decoder=AttnDecoderRNN(hidden_size=config["hidden_size"],output_size=config["vocab_size"],dropout_p=0.1).to(device)
model=model(encoder,decoder,config)

model.trainIters(train_pairs=pairs,n_iters=2)


