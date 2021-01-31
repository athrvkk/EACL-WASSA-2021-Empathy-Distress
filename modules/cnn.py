# File contatins Class CNN
# File: cnn.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

from utils.utils import Utils
from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, Concatenate, GlobalMaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


class CNN():

    def __init__(self, activation):
        self.activation = activation

    def build(self, input_length, embedding_matrix):
        """ Function to build CNN as base model.
        """
        input = Input(shape=(input_length,))
        x = Embedding(input_dim=embedding_matrix.shape[0], 
                      output_dim=embedding_matrix.shape[1], 
                      weights=[embedding_matrix], trainable=False)(input)
        
        
        x1 = Conv1D(filters=128, kernel_size=3, padding="same", activation=self.activation, kernel_regularizer=l2(0.001))(x)
        x1 = GlobalAveragePooling1D()(x1)
        x1 = Dropout(0.2)(x1)

        x2 = Conv1D(filters=128, kernel_size=4, padding="same", activation=self.activation, kernel_regularizer=l2(0.001))(x)
        x2 = GlobalAveragePooling1D()(x2)
        x2 = Dropout(0.2)(x2)
        
        x3 = Conv1D(filters=128, kernel_size=5, padding="same", activation=self.activation, kernel_regularizer=l2(0.001))(x)
        x3 = GlobalAveragePooling1D()(x3)
        x3 = Dropout(0.2)(x3)

        conc = Concatenate(axis=1)([x1, x2, x3])
        conc = Dropout(0.2)(conc)

        dense = Dense(128,  activation=self.activation, kernel_regularizer=l2(0.001))(conc)
        out = Dropout(0.2)(dense)

        return Model(inputs=input, outputs=out)


