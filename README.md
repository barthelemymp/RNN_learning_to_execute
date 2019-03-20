README


# PRESENTATION

This github repository aims at exploring a few technics of seq2seq. To do so we have focused on the simple problem of translating written numbers to integer.These problem is quite useful to introduce the mains problematics of traduction but in a simpler way. Indeed one main problem of the classical traduction is the diffuclty to evaluate the quality of a traduction. In numbers translation, only one answer is correct which solve this issue.


## STRUCTURE

- baseline_Keras: the baseline we used as a start (in Keras).
- data_npy: where we store the dataset (before tokenization)
- result: where we store the results (the curves we used during the presentation)
- saved_models: where we save the model with the best results -> load first_deco.pt and first_enco.pt to get 97% of correct translations



## CODES
 
* [model.py] (https://github.com/barthelemymp/RNN_learning_to_execute/blob/master/model.py)

This python code define the different part of our neural network. There are three classes:
- Encoder : this class define an encoder which include an embbedding layer and a GRU.
- Decoder : this class define the decoder which include an embedding, an attention mecanism, a dropout layer and GRU.
- Model : this class needs an encoder a decoder and a few parameter to be initialized. It contains all the functions that use the encoder and the decoder . La fonction trainIters entraine le model. The decode function predict the number translation (once the model has been trained)

* [train.py] (https://github.com/barthelemymp/RNN_learning_to_execute/blob/master/train.py) : run this code to train the model and save it

* [decode.py] (https://github.com/barthelemymp/RNN_learning_to_execute/blob/master/decode.py) : code to make prediction

* [data_creation.py] (https://github.com/barthelemymp/RNN_learning_to_execute/blob/master/data_creation.py) : create the dataset

* [utils.py] (https://github.com/barthelemymp/RNN_learning_to_execute/blob/master/utils.py) : gathers useful functions for this problem



