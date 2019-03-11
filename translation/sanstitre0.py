# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:12:24 2019


encodage des nombres

@author: barthelemy
"""

from __future__ import print_function
import torch
import torch.nn as nn
import os
import numpy as np

import numpy as np
from tokenization import tokenize
from tokenization import build_vocabulary_token
from tokenization import vectorize_corpus
import csv

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

vocab_size = len(shared_vocab)
config = {
        'dropout': 0.2,
        'vocab_size': vocab_size,
        'num_layers': 1,
        'embsize': 32,
        'dim_recurrent':50,
    }


class Encoder1(nn.Module):

    def __init__(self, config):
        super(Encoder1, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['embsize'])
        self.drop = nn.Dropout(config['dropout'])
        self.rnn = nn.LSTM(input_size = config['embsize'], 
                           hidden_size = config['dim_recurrent'],
                           num_layers = config['num_layers'])
#        self.gru = nn.gru(input_size = config['dim_input'], 
#                           hidden_size = config['dim_recurrent'],
#                           num_layers = config['num_layers'])
        self.dense = nn.Linear(config['dim_recurrent'],config['vocab_size'])

    def forward(self, x):
        layer_embeded = self.embed(x)
        layer_drop = self.drop(layer_embeded)
        layer_rnn = self.rnn(layer_drop, hidden)[0]
        out = self.dense(F.softmax(layer_rnn))
        return out
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(config['num_layers'], 1, config['dim_recurrent']),
                torch.zeros(config['num_layers'], 1, config['dim_recurrent']))
    


model = Encoder1(config)
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    



with torch.no_grad():
    inputs = X_train[0]
    tag_scores = model(inputs)
    print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for i in range(len(X_train)):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = X_train[i]
        targets =  Y_train[i]

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = X_train[0]
    tag_scores = model(inputs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#        
#        print(x.shape)
#        x = x.unsqueeze(1)
#        print(x.shape)
#        output, _ = self.lstm(x)
#        print(output.shape)
#        output = output.squeeze(1)
#        print(output.shape)
#        output = output.narrow(0, output.size(0)-1,1)
#        print(output.shape)
#        
#        return self.fc_o2y(F.softmax(output))
#
#
#
#
#
#simple_seq2seq = Sequential()
#simple_seq2seq.add(Embedding(vocab_size, 32, input_length=max_length))
#simple_seq2seq.add(Dropout(0.2))
#simple_seq2seq.add(GRU(256, return_sequences=True))
#simple_seq2seq.add(Dense(vocab_size, activation='softmax'))