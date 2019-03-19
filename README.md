# RNN_learning_to_execute

<<<<<<< HEAD

##################################################################################################
###                                        INTRODUCTION                                       ####
##################################################################################################

This github repository aims at exploring a few technics of seq to seq.To do so we have focused on the simple problem of translating written numbers to integer.These problem is quite useful to introduce the mains problematics of traduction but in a simpler way. Indeed one main problem of the classical traduction is the diffuclty to evaluate the quality of a traduction. In numbers translation, only one answer is correct which solve this issue.


###Structure





# Files 
model.py
This python code define the different part of our neural network. There are three classes:
- Encoder : this class define an encoder which include an embbedding layer and a GRU.
- Decoder : this class define the decoder which include an embedding, an attention mecanism, a dropout layer and GRU.
- Model : this class needs an encoder a decoder and a few parameter to be initialized. It contains all the functions that use the encoder and the decoder . La fonction trainIters entraine le model. The decode function predict the number translation (once the model has been trained)

train.py
run this code to train the model and save it

decode.py
code to make prediction









=======
Problem : building a translation model from french phrases describing numbers to the corresponding numeric representation (base 10).

1. tokenization.py
>>>>>>> 594f218982cc1f28b0930cd30ee5b8e2a410c7d8
