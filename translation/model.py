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
from tokenization import tokenize
from tokenization import build_vocabulary_token
from tokenization import vectorize_corpus
import csv
import time


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



use_gpu = torch.cuda.is_available()
def gpu(tensor, gpu=use_gpu):
    if gpu:
        return tensor.cuda()
    else:
        return tensor

vocab_size = len(shared_vocab)
batch_size=32

config ={
        'dropout': 0.2,
        'vocab_size': vocab_size,
        'num_layers': 1,
        'embsize': 32,
        'dim_recurrent': 256,
        'batch_size':batch_size,
        'hidden_size': 256,
        'max_length': 20
    }


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.param = [input_size,hidden_size]
        self.embedding = nn.Embedding(input_size,hidden_size)#(self.input_size, self.hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #input2 = torch.tensor(input)
        #print('avant call', input.dtype)
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
    
    
    
    
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)



class model(nn.Module):
    def __init__(self,encoder,decoder, config,  criterion = nn.NLLLoss(),train_pairs = pairs):
        super(model, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_pairs = train_pairs
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=0.01)
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=0.01)
        self.criterion = criterion
        self.config = config


    def train(self,input_tensor, target_tensor):
    
        teacher_forcing_ratio = 0.5

        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.config['max_length'], encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[GO_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length

    def trainIters(self,n_iters,epochs=5, print_every=100, plot_every=100, learning_rate=0.01,n_evaluate=1000):
     
        all_losses=[]
        all_test_losses=[]

        start = time.time()
        print_loss_total = 0.  # Reset every print_every
        plot_loss_total = 0. # Reset every plot_every

        training_pairs = [random.choice(self.train_pairs)for i in range(n_iters)]


        for epoch in range(0,epochs):

            for iter in range(1, n_iters + 1):
                training_pair = training_pairs[iter - 1]
                input_tensor = torch.tensor(training_pair[0])
                target_tensor = torch.tensor(training_pair[1])

                loss = self.train(input_tensor, target_tensor)
                print_loss_total += loss
                plot_loss_total += loss

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print("epoch :" + str(epoch) + " iter : " + str(iter))
                    print( print_loss_avg)


                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_loss_total = 0.
                    all_losses+=[plot_loss_avg]

            loss_PATH="losses_epoch"+str(epoch)
            np.save(loss_PATH,np.array(all_losses))

            enco_PATH="encoder_epoch"+str(epoch)
            deco_PATH="decoder_epoch"+str(epoch)
            torch.save(encoder,enco_PATH)
            torch.save(encoder,deco_PATH)

        torch.save(encoder,"second_enco")
        torch.save(encoder,"second_deco")       
        np.save('all_losses',np.array(all_losses))


    def decode(self,input_tensor):

        max_length=self.config['max_length']
        input_length = input_tensor.size(0)
        target_length = self.config['max_length']

        encoder_hidden = self.encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, self.encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[GO_token]], device=device)
        decoder_hidden = encoder_hidden

        solu=torch.zeros(max_length)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()
            solu[di]=decoder_input


        return solu
        
    
    
    
    
    
