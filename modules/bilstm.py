# File contatins Class BiLSTM
# File: bilstm.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

from utils.utils import Utils
from tensorflow.keras.layers import Input, Embedding, Dense, Bidirectional, LSTM, GlobalMaxPool1D, GlobalAveragePooling1, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


class BiLSTM():

    def __init__(self, activation, kr_initializer, kr_rate):
        self.activation = activation

    def build(self, input_length, embedding_matrix):
        """ Function to build CNN as base model.
        """
        input = Input(shape=(input_length,))
        embedding = Embedding(input_dim=embedding_matrix.shape[0], 
                              output_dim=embedding_matrix.shape[1], 
                              weights=[embedding_matrix], trainable=False)(input)
        lstm = Bidirectional(LSTM(128, dropout=0.3, return_sequences=True))(embedding)
        lstm_pool = GlobalMaxPool1D()(lstm)
        dense = Dense(128,  activation=self.activation, kernel_initializer=self.kr_initializer, kernel_regularizer=l2(self.kr_rate))(lstm_pool)
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




            



