import numpy as np
from tokenization import tokenize
from tokenization import build_vocabulary_token
from tokenization import vectorize_corpus
from tokenization import to_sentence
import csv
import torch
from numberseqseq import EncoderRNN
from numberseqseq import AttnDecoderRNN

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

encoder=torch.load('saved_models/first_enco.pt')
decoder=torch.load('saved_models/first_deco.pt')

MAX_LENGTH=20
GO_token = 1
EOS_token = 2

def decode(input_tensor,encoder,decoder):

    max_length=MAX_LENGTH
    input_length = input_tensor.size(0)
    target_length = MAX_LENGTH

    encoder_hidden = encoder.initHidden()
    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[GO_token]])
    decoder_hidden = encoder_hidden

    solu=torch.zeros(max_length)

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        solu[di]=decoder_input


    return solu

score=0

for i in range(1000):
    x_test=torch.tensor(pairs[i][0])
    y_test=torch.tensor(pairs[i][1])
    x_pred=decode(x_test,encoder,decoder)
    x_pred=x_pred.numpy()
    num_pred=to_sentence(x_pred,rev_shared_vocab)
    if int(num_pred)==int(num_val[i]):
    	score+=1
    print(int(num_pred),int(num_val[i]))
    print(int(num_pred)-int(num_val[i]))

print(score)
print(score/1000)


