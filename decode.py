''' Decodes a sentence and transform it into numbers  '''

import torch
from model import EncoderRNN
from model import AttnDecoderRNN
from model import model

import numpy as np
from utils import tokenize
from utils import build_vocabulary_token
from utils import vectorize_corpus
from utils import to_sentence

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


enco=torch.load('saved_models/first_enco.pt')
deco=torch.load('saved_models/first_deco.pt')

model = model(encoder=enco, decoder=deco, config=config)
print(model.decode((torch.tensor(X_test[0], dtype=torch.long, device=device).view(-1, 1))))
print(to_sentence(Y_test[0],rev_shared_vocab))