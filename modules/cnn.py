# file contatins Class CNN
# File: cnn.py
# Author: Atharva Kulkarni




from tensorflow.keras.layers import Input, Embedding, Dense, Conv1D, Concatenate
from tensorflow.keras.layers import AveragePooling1D, MaxPooling1D, GlobalMaxPool1D, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import Model


class CNN():

    def build(self, input_length, embedding_matrix):
        """ Function to build CNN as base model.
        """
        input = Input(shape=(input_length,))
        x = Embedding(input_dim=embedding_matrix.shape[0], 
                      output_dim=embedding_matrix.shape[1], 
                      weights=[embedding_matrix], trainable=False)(input)
        
        
        x1 = Conv1D(filters=128, kernel_size=3, padding="same", kernel_regularizer=l2(0.001))(x)
        x1 = GlobalMaxPool1D()(x1)
        x1 = Dropout(0.2)(x1)

        x2 = Conv1D(filters=128, kernel_size=4, padding="same", kernel_regularizer=l2(0.001))(x)
        x2 = GlobalMaxPool1D()(x2)
        x2 = Dropout(0.2)(x2)
        
        x3 = Conv1D(filters=128, kernel_size=5, padding="same", kernel_regularizer=l2(0.001))(x)
        x3 = GlobalMaxPool1D()(x3)
        x3 = Dropout(0.2)(x3)

        x = Concatenate(axis=1)([x1, x2, x3])
        x = Dropout(0.2)(x)

        x = Dense(128, activation="relu", kernel_regularizer=l2(0.001))(x)
        x = Dropout(0.2)(x)

        return Model(inputs=input, outputs=x)

