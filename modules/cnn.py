# File contatins Class CNN
# File: cnn.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

from utils.utils import Utils
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


class CNN():

    def __init__(self, activation, kr_initializer, kr_rate):
        self.activation = activation

    def build(self, input_length, embedding_matrix):
        """ Function to build CNN as base model.
        """
        input = Input(shape=(input_length,))
        x = Embedding(input_dim=embedding_matrix.shape[0], 
                      output_dim=embedding_matrix.shape[1], 
                      weights=[embedding_matrix], trainable=False)(input)
        
        
        x1 = Conv1D(filters=128, kernel_size=3, padding="same", activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x)
        x1 = GlobalAveragePooling1D()(x1)
        x1 = Dropout(0.2)(x1)

        x2 = Conv1D(filters=128, kernel_size=4, padding="same", activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x)
        x2 = GlobalAveragePooling1D()(x2)
        x2 = Dropout(0.2)(x2)
        
        x3 = Conv1D(filters=128, kernel_size=5, padding="same", activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(x)
        x3 = GlobalAveragePooling1D()(x3)
        x3 = Dropout(0.2)(x3)

        conc = Concatenate(axis=1)([x1, x2, x3])
        conc = Dropout(0.2)(conc)

        dense = Dense(128,  activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(conc)
        out = Dropout(0.2)(dense)

        return Model(inputs=input, outputs=out)



    def prepare_input(self, utils_obj, corpus, maxlen, padding_type, truncating_type, mode):
        if mode =="train":
            preprared_corpus, myTokenizer = utils_obj.tokenize_and_pad(corpus, 
                                                                       maxlen=maxlen,  
                                                                       padding_type=padding_type,
                                                                       truncating_type=truncating_type, 
                                                                       mode="train")
            return preprared_corpus, myTokenizer
        else:
            preprared_corpus, _ = utils_obj.tokenize_and_pad(corpus, 
                                                             maxlen=maxlen, 
                                                             padding_type=padding_type,
                                                             truncating_type=truncating_type, 
                                                             mode="test")
            return preprared_corpus








