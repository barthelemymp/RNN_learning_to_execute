# PRESENTATION

This github repository aims at exploring a few technics of seq2seq. To do so we have focused on the simple problem of translating written numbers to integer. This problem is quite useful to introduce the mains problematics of translation but in a simpler way. Indeed one main problem of the classical traduction is the diffuclty to evaluate the quality of a traduction. In numbers translation, only one answer is correct which solve this issue.
Please find a short presentation of the project and the NN architecture [here](https://docs.google.com/presentation/d/1nRMQdYdciJA7pyb-NW1MKJdEfhI2c8zlL1iNgPIr8X4/edit?usp=sharing_eip&ts=5c8fe66c) and discover the project with the [Jupyter notebook](numberseq2seq.ipynb) we created.

## STRUCTURE

- [baseline_Keras](baseline_Keras): the baseline we used as a start (in Keras).
- [data_npy](data_npy): where we store the dataset (before tokenization)
- [result](result): where we store the curves we used during the presentation
- [saved_models](saved_models): where we save the model with the best results -> load first_deco.pt and first_enco.pt to get 97% of correct translations


## CODES

* [The notebook of the project](numberseq2seq.ipynb)
We provide a Jupyter notebook that helps to understand how we built, trained and tested our model. The following files correspond to the global architecture of our project.
 
* [model.py](model.py)

This python code define the different part of our neural network. There are three classes:
-  Encoder : this class define an encoder which include an embbedding layer and a GRU.
-  Decoder : this class define the decoder which include an embedding, an attention mecanism, a dropout layer and GRU.
-  Model : this class needs an encoder a decoder and a few parameter to be initialized. It contains all the functions that use the encoder and the decoder . La fonction trainIters entraine le model. The decode function predict the number translation (once the model has been trained)

* [train.py](train.py)
run this code to train the model and save it

* [decode.py](decode.py)
code to make prediction

* [data_creation.py](data_creation.py)
create the dataset

* [utils.py](utils.py)
gathers useful functions for this problem



