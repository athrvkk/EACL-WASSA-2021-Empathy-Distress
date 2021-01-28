# file contatins Class BertUnit, implementation of BERT for regression
# File: BertUnit.py
# Author: Atharva Kulkarni

import time
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split


from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, GlobalMaxPool1D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

from transformers import TFBertModel, BertTokenizer
from  transformers import TFDistilBertModel, DistilBertTokenizer
from  transformers import TFRobertaModel, RobertaTokenizer

import gc
import warnings
warnings.filterwarnings('ignore')





class BertUnit():
    """ Implementation of BERT for regression """


    def __init__(self, bert_model="BERT", output_hidden_states=False):
        """ Constructor to initialize Deep learning models
        @param model (str): the model to be used (DNN, CNN, BiLSTM, BERT, DistilBERT, RoBERTa).
        @param output_hidden_states (bool): Whether to output the hidden states of BERT.
        """
        if bert_model == "BERT":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            self.bert = TFBertModel.from_pretrained("bert-base-uncased", output_hidden_states=output_hidden_states)

        elif bert_model == "DistilBERT":
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
            self.bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased", output_hidden_states=output_hidden_states)

        elif bert_model == "RoBERTa":
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
            self.bert = TFRobertaModel.from_pretrained("roberta-base", output_hidden_states=output_hidden_states)





    def build(self, input_ids_dim, attention_mask_dim, fc1_dim, fc2_dim, d1, d2):
        """ Function to define the forward pass of the model.
        @param input (tensor): input to the model.
        """
        input_ids = Input(shape=(input_ids_dim,), name='input_ids', dtype="int32")
        attention_masks = Input(shape=(attention_mask_dim,), name='attention_mask', dtype="int32")
        x = self.bert(input_ids, attention_mask=attention_masks)[0]
        x = GlobalMaxPool1D()(x)
        x = BatchNormalization()(x)
        x = Dense(fc1_dim, activation="relu")(x)
        x = Dropout(d1)(x)
        x = Dense(fc2_dim, activation="relu")(x)
        x = Dropout(d2)(x)
        score = Dense(1)(x)

        self.model = Model(inputs=[input_ids, attention_masks], outputs=score)
        self.model.layers[2].trainable = False
        self.model.compile(optimizer="adam", loss="mse")
        self.model.summary()





    def train(self, x_train1, x_train2, train_score, x_val1, x_val2, val_score, epochs=200, batch_size=32):
        history = self.model.fit(x=[x_train1, x_train2],
                                 y=train_score,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 verbose=1,
                                 validation_data=([x_val1, x_val2], val_score))
        return history





    def prepare_bert_input(self, corpus, max_len=200):
        """ Function to prepare input data for BERT.
        @param corpus (list): Dataset to be processed.
        @param max_len (int): maximum length of input post. Texts longer that max_len will be truncated, while shorter than max_len will be padded.
        """
        input_ids = []
        attention_masks = []
        
        for record in corpus:
            encoded_text = self.tokenizer.encode_plus(text=record,
                                                      add_special_tokens=True,
                                                      return_attention_mask=True,
                                                      max_length=max_len,
                                                      pad_to_max_length=True,
                                                      truncation=True)
            input_ids.append(encoded_text.get("input_ids"))
            attention_masks.append(encoded_text.get("attention_mask"))

        return np.array(input_ids, dtype="int32"), np.array(attention_masks, dtype="int32")
        




    
