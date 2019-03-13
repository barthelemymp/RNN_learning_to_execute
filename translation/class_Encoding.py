import imp
import numpy as np

import torch.optim as optim

#from __future__ import print_function
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

class Encoder1(nn.Module):

    def __init__(self, config, model=None, n_iter=10, batch_size=32,l2=0.0,learning_rate=0.01, random_state=None, use_cuda=False):
        super(Encoder1, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config['vocab_size'], config['embsize'])
        self.drop = nn.Dropout(config['dropout'])
        self._use_cuda = use_cuda
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
        self._model = model
        self._batch_size = batch_size
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._l2 = l2
        self._optimizer = None
        self._loss_func = None


    def _initialize(self):
        if self._model is None:
            self._model = gpu(Encoder1(config), gpu=self._use_cuda)

        self._optimizer = optim.Adam(
                self._net.parameters(),
                lr=self._learning_rate,
                weight_decay=self._l2
            )
        
        self._loss_func = nn.NLLLoss()
    
    @property
    def _initialized(self):
        return self._optimizer is not None
    

    def fit(self, X_train, Y_train, verbose=True):
        
        if not self._initialized:
            self._initialize()

        idx=np.arange(X_train.shape[0])
        
        all_losses=[]

        for epoch in range(self._n_iter):  # again, normally you would NOT do 300 epochs, it is toy data
            np.random.shuffle(idx)
            X_train=X_train[idx]
            Y_train=Y_train[idx]

            current_batch=0
            current_loss=0.0
            for i in range(len(X_train)//self._batch_size):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self._model.zero_grad()
                self._optimizer.zero_grad()

                batch_x=X_train[current_batch:current_batch+self._batch_size]
                batch_y=torch.tensor(Y_train[current_batch:current_batch+self._batch_size],dtype=torch.int64)
                current_batch+=self._batch_size

                # Step 3. Run our forward pass.
                batch_pred = self._model(batch_x)


                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = self._loss_function(batch_pred, batch_y)
                current_loss += loss.item()
                loss.backward()
                self._optimizer.step()


                if i % 100 == 0 :
                    with torch.no_grad():
                        self._model.train(False)
                        #batch_pred, batch_y = test_eval()
                        #f1 = f1_score(batch_y,batch_pred, average="weighted")
                        #precision = precision_score(batch_y,batch_pred,average="weighted")
                        all_losses.append(current_loss/100)
                        print(loss.item(),'\titeration:', i, '\tepoch', epoch)
                        self._model.train(True)
                        current_loss=0.
                
        
    

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
        return (torch.zeros(config['num_layers'], self._batch_size, config['dim_recurrent']),
                torch.zeros(config['num_layers'], self._batch_size, config['dim_recurrent']))

    def test(self,X_val, Y_val):
        self._model.train(False)
        self._model = gpu(Encoder1(config))
        predictions = self._model(X_val)
        loss = self._loss_func(Y_val, predictions)
        return loss.data.item()
