from tokenization import generate_translations
from tokenization import tokenize
from sklearn.model_selection import train_test_split
from tokenization import build_vocabulary
import numpy as np

PAD, GO, EOS, UNK = START_VOCAB = ['_PAD', '_GO', '_EOS', '_UNK']
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

print("C'est tipar")

numbers, french_numbers = generate_translations(
    low=1, high=int(1e6) - 1, exhaustive=40000, random_seed=0)

num_train, num_dev, fr_train, fr_dev = train_test_split(numbers, french_numbers, test_size=0.5, random_state=0)

num_val, num_test, fr_val, fr_test = train_test_split(num_dev, fr_dev, test_size=0.5, random_state=0)

fr_vocab, rev_fr_vocab = build_vocabulary(fr_train)
num_vocab, rev_num_vocab = build_vocabulary(num_train)

np.save('data_npy/fr_train',fr_train)
np.save('data_npy/num_train',num_train)
np.save('data_npy/fr_val',fr_val)
np.save('data_npy/num_val',num_val)
np.save('data_npy/fr_test',fr_test)
np.save('data_npy/num_test',num_test)
np.save('data_npy/fr_vocab',fr_vocab)
np.save('data_npy/rev_fr_vocab',rev_fr_vocab)
np.save('data_npy/num_vocab',num_vocab)
np.save('data_npy/rev_num_vocab',rev_num_vocab)