import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

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

"""
Blog post:
Taming LSTMs: Variable-sized mini-batches and why PyTorch is good for your health:
https://medium.com/@_willfalcon/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
"""

use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor

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


print(shared_vocab)

class LSTM(nn.Module):
    def __init__(self, max_len = 20, nb_lstm_units=10, nb_lstm_layers=2, embedding_dim=32, batch_size=4, vocab=shared_vocab, dropout=0.2, gpu=True):
        super(LSTM, self).__init__()
        self.vocab = vocab
        self.tags = vocab
        self.dropout = dropout
        self.nb_lstm_layers = nb_lstm_layers
        self.nb_lstm_units = nb_lstm_units
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.on_gpu = gpu
        # don't count the padding tag for the classifier output
        self.nb_tags = len(self.tags) - 1
        self.length = max_len
        # when the model is bidirectional we double the output dimension
        #self.lstm

        # build actual NN
        self.__build_model()

    def __build_model(self):
        # build embedding layer first
        nb_vocab_words = len(self.vocab)

        # whenever the embedding sees the padding index it'll make the whole vector zeros
        padding_idx = self.vocab['_PAD']

        self.word_embedding = nn.Embedding(
            num_embeddings=nb_vocab_words,
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx
        )
        self.drop = nn.Dropout(self.dropout)
        # design LSTM
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            num_layers=self.nb_lstm_layers,
            batch_first=True
        )

        # output layer which projects back to tag space
        self.hidden_to_tag = nn.Linear(self.nb_lstm_units, self.nb_tags)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        if self.on_gpu:
            hidden_a = hidden_a.cuda()
            hidden_b = hidden_b.cuda()

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)

        return (hidden_a, hidden_b)

    def forward(self, X):
        # reset the LSTM hidden state. Must be done before you run a new batch. Otherwise the LSTM will treat
        # a new batch as a continuation of a sequence
        self.hidden = self.init_hidden()

        batch_size, seq_len = X.shape

        # ---------------------
        # 1. embed the input
        # Dim transformation: (batch_size, seq_len, 1) -> (batch_size, seq_len, embedding_dim)
        X=torch.LongTensor(X) #batch_size, seq_len
        X = self.word_embedding(X) #(batch_size, seq_len, embedding_dim)
        X = self.drop(X) #(batch_size, seq_len, embedding_dim)
        # ---------------------
        # 2. Run through RNN
        # TRICK 2 ********************************
        # Dim transformation: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, nb_lstm_units)

        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM
        X = torch.nn.utils.rnn.pad_sequence(X) #seq_len, batch_size, embedding_dim
        X=X.transpose(0,1) #(batch_size, seq_len, embedding_dim)
        # now run through LSTM
        X, self.hidden = self.lstm(X, self.hidden) #(batch_size, seq_len, nb_lstm_units)
        # undo the packing operation
        X = torch.nn.utils.rnn.pad_sequence(X) #(seq_len, batch_size, nb_lstm_units)
        
        # ---------------------
        # 3. Project to tag space
        # Dim transformation: (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)

        # this one is a bit tricky as well. First we need to reshape the data so it goes into the linear layer
        #X = X.contiguous()
        #X = X.view(-1, X.shape[2])

        # run through actual linear layer
        X = self.hidden_to_tag(X) #seq_len,batch_size, nb_tags
        # ---------------------
        # 4. Create softmax activations bc we're doing classification
        # Dim transformation: (batch_size * seq_len, nb_lstm_units) -> (batch_size, seq_len, nb_tags)
        X = F.log_softmax(X, dim=2)
        # I like to reshape for mental sanity so we're back to (batch_size, seq_len, nb_tags)
        X=X.transpose(0, 1)
        X = X.view(batch_size, seq_len, self.nb_tags) #batch_size, seq_len, nb_tags
        Y_hat = X
        return Y_hat

    def loss(self, Y_hat, Y):
        # TRICK 3 ********************************
        # before we calculate the negative log likelihood, we need to mask out the activations
        # this means we don't want to take into account padded items in the output vector
        # simplest way to think about this is to flatten ALL sequences into a REALLY long sequence
        # and calculate the loss on that.

        # flatten all the labels
        Y = Y.contiguous()
        Y = Y.view(-1)
        # flatten all predictions
        Y_hat = Y_hat.contiguous() #batch_size, seq_len, nb_tags
        # Dim transformation: (batch_size, seq_len, nb_tags) -> (batch_size * seq_len, nb_tags) 
        print(Y_hat)
        print(Y_hat.shape)
        Y_hat = Y_hat.view(-1, self.nb_tags)
        print(Y_hat)
        print(Y_hat.shape)
        #Y_hat = [mask,Y_hat]
        # create a mask by filtering out all tokens that ARE NOT the padding token
        tag_pad_token = self.tags['_PAD']
        mask = (Y > tag_pad_token).float()
        # count how many tokens we have
        nb_tokens = int(torch.sum(mask))
        #Y_pred = 
        # pick the values for the label and zero out the rest with the mask
        Y_hat = Y_hat[range(Y_hat.shape[0]), Y] * mask
        # compute cross entropy loss which ignores all <PAD> tokens
        ce_loss = -torch.sum(Y_hat) / nb_tokens
        return ce_loss



batch_size=4

model = LSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)    

start=time.time()
current_loss= 0.
all_losses=[]


# with torch.no_grad():

#     inputs = np.concatenate((np.expand_dims(X_train[0],axis=0),np.expand_dims(X_train[1],axis=0)),axis=0)
#     tag_scores = model(inputs)


idx=np.arange(X_train.shape[0])

for epoch in range(5):  # again, normally you would NOT do 300 epochs, it is toy data
     np.random.shuffle(idx)
     X_train=X_train[idx]
     Y_train=Y_train[idx]
     print("je suis la")
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
         #mask = masking(batch_x)
         batch_y=torch.tensor(Y_train[current_batch:current_batch+batch_size],dtype=torch.int64)
         current_batch+=batch_size

         # Step 3. Run our forward pass.
         batch_pred = model(batch_x)

         # Step 4. Compute the loss, gradients, and update the parameters by
         #  calling optimizer.step()
         # Use the mask variable to avoid the effect of padding on loss
         # function calculations
         loss = model.loss(batch_pred, batch_y)
         #loss = (loss * mask.shape[1] / torch.sum(mask,dim=1))
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


np.save('result/losses_1.npy', all_losses)

torch.save(model.state_dict(), 'LSTM_mask_trained.pt')

x_val=X_train[0:batch_size]
y_val=Y_train[0:batch_size]
print("y_val", y_val)
y_pred=model(x_val)
#batch_size, tag, seq_len
y_pred = y_pred.transpose(1,2)
print(y_pred)
print(y_pred.shape)
_,y_pred=torch.max(y_pred,dim=1)
print(y_pred)
print(y_pred.shape)

print("y_pred", y_pred)
print("y_val", y_val)
y_pred = torch.tensor(y_pred,dtype=torch.int64)
print(y_pred)
print(y_pred.shape)
loss = model.loss(y_pred, y_val)
print("loss",loss)

