# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:12:24 2019


encodage des nombres

@author: barthelemy
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
#import matplotlib.pyplot as plt

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

# Create the shared dictionary
tokenized_fr_train = [tokenize(s, word_level=True) for s in fr_train]
tokenized_num_train = [tokenize(s, word_level=False) for s in num_train]
shared_vocab, rev_shared_vocab = build_vocabulary_token(tokenized_fr_train+tokenized_num_train)

# Create the training, evaluating and testing sets
X_train, Y_train = vectorize_corpus(fr_train, num_train, shared_vocab,word_level_target=False)
X_val, Y_val = vectorize_corpus(fr_val, num_val, shared_vocab,word_level_target=False)
X_test, Y_test = vectorize_corpus(fr_test, num_test, shared_vocab,word_level_target=False)

use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor

vocab_size = len(shared_vocab)
batch_size=8

config ={
        'dropout': 0.2,
        'vocab_size': vocab_size,
        'num_layers': 1,
        'embsize': 32,
        'dim_recurrent': 50,
        'num_layers': 1,
        'num_layers_int': 1,
        'batch_size':batch_size
    }


# config_tensor = {
#         'dropout': torch.tensor(0.2),
#         'vocab_size': torch.tensor(vocab_size),
#         'num_layers': torch.tensor(1),
#         'embsize': torch.tensor(32),
#         'dim_recurrent':torch.tensor(50),
#         'num_layers':torch.FloatTensor(1),
#         'num_layers_int':torch.IntTensor(1)
#     }


class Encoder1(nn.Module):

    def __init__(self, config):
        super(Encoder1, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['embsize'])
        self.drop = nn.Dropout(config['dropout'])

        self.rnn = nn.LSTM(input_size = config['embsize'], 
                           hidden_size = config['dim_recurrent'])


        # BART
        #self.rnn = nn.LSTM(input_size = config['embsize'], 
        #                   hidden_size = config['dim_recurrent'],
        #                   num_layers = config['num_layers'])


#        self.gru = nn.gru(input_size = config['dim_input'], 
#                           hidden_size = config['dim_recurrent'],
#                           num_layers = config['num_layers'])
        self.dense = nn.Linear(config['dim_recurrent'],config['vocab_size'])
        self.hidden = self.init_hidden()

    def forward(self, x):

        # BATCH VERSION
        x2=torch.LongTensor(x)

        layer_embeded = self.embed(x2)

        layer_drop = self.drop(layer_embeded)

        layer_drop=layer_drop.transpose(0,1)

        layer_rnn = self.rnn(layer_drop, self.hidden)[0]

        layer_dense = self.dense(layer_rnn)

        out=F.softmax(layer_dense,dim=2) #[20,2,40]

        out3=out.transpose(0, 1)
        out4=out3.transpose(1,2)

        return out4
    
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(config['num_layers'], batch_size, config['dim_recurrent']),
                torch.zeros(config['num_layers'], batch_size, config['dim_recurrent']))


model = gpu(Encoder1(config))
loss_function = nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)    

start=time.time()
current_loss= 0.
all_losses=[]


# with torch.no_grad():

#     inputs = np.concatenate((np.expand_dims(X_train[0],axis=0),np.expand_dims(X_train[1],axis=0)),axis=0)
#     tag_scores = model(inputs)


idx=np.arange(X_train.shape[0])

for epoch in range(1):  # again, normally you would NOT do 300 epochs, it is toy data
    np.random.shuffle(idx)
    X_train=X_train[idx]
    Y_train=Y_train[idx]

    current_batch=0
    for i in range(len(X_train)//batch_size):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()


        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        #model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.

        batch_x=X_train[current_batch:current_batch+batch_size]
        batch_y=torch.tensor(Y_train[current_batch:current_batch+batch_size],dtype=torch.int64)
        current_batch+=batch_size

        # Step 3. Run our forward pass.
        batch_pred = model(batch_x)


        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(batch_pred, batch_y)
        loss.backward()
        optimizer.step()

        current_loss += loss.item()


        if i % 100 == 0 :
            with torch.no_grad():
                model.train(False)
                #batch_pred, batch_y = test_eval()
                #f1 = f1_score(batch_y,batch_pred, average="weighted")
                #precision = precision_score(batch_y,batch_pred,average="weighted")
                all_losses.append(current_loss/100)
                print(loss.item(),'\titeration:', i, '\tepoch', epoch)
                model.train(True)
                current_loss=0.



np.save(all_losses,'all_losses')

# print(model(X_val[0]))
# print(Y_val[0])

# See what the scores are after training
# with torch.no_grad():
#     inputs = X_train[0]
#     tag_scores = model(inputs)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
