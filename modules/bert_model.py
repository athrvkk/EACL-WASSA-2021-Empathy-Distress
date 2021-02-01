# File contatins Class BertModel
# File: bert_model.py
# Author: Atharva Kulkarni

import sys
sys.path.append('../')

from transformers import TFBertModel, BertConfig, BertTokenizer, TFAutoModel
from transformers import TFDistilBertModel, DistilBertConfig, DistilBertModel, DistilBertTokenizer
from transformers import TFRobertaModel, RobertaConfig, RobertaModel, RobertaTokenizer

from tensorflow.keras.layers import Input, Dense, Dropout, Conv1D, GlobalMaxPool1D, GlobalAveragePooling1D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model



class BertModel():
    """ Implementation of BERT for regression """


    def __init__(self, bert_model="BERT", activation="relu", output_hidden_states=False):
        """ Constructor to initialize Deep learning models
        @param model (str): the model to be used (DNN, CNN, BiLSTM, BERT, DistilBERT, RoBERTa).
        @param output_hidden_states (bool): Whether to output the hidden states of BERT.
        """
        self.activation = activation

        if bert_model == "BERT":
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=output_hidden_states)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            self.bert = TFBertModel.from_pretrained("bert-base-uncased", config=config)

        elif bert_model == "DistilBERT":
            config = DistilBertConfig.from_pretrained("distilbert-base-uncased", output_hidden_states=output_hidden_states)
            self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
            self.bert = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

        elif bert_model == "RoBERTa":
            config = RobertaConfig.from_pretrained("roberta-base", output_hidden_states=output_hidden_states)
            self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base", do_lower_case=True)
            self.bert = TFRobertaModel.from_pretrained("roberta-base", config=config)





    def prepare_input(self, corpus, maxlen):
        """ Function to prepare input data for BERT.
        @param corpus (list): Dataset to be processed.
        @param max_len (int): maximum length of input post. Texts longer that max_len will be truncated, while shorter than max_len will be padded.
        """
        input_ids = []
        attention_mask = []

        for record in corpus:
            encoded_text = self.tokenizer.encode_plus(text=record,
                                                      add_special_tokens=True,
                                                      return_attention_mask=True,
                                                      max_length=maxlen,
                                                      pad_to_max_length=True,
                                                      truncation=True)
            input_ids.append(encoded_text.get("input_ids"))
            attention_mask.append(encoded_text.get("attention_mask"))

        return [np.array(input_ids, dtype="int32"), np.array(attention_mask, dtype="int32")]





    def build(self, input_length):
        """ Function to define the forward pass of the model.
        @param input (tensor): input to the model.
        """
        input_ids = Input(shape=(input_length,), name='input_ids', dtype="int32")
        attention_mask = Input(shape=(input_length,), name='attention_mask', dtype="int32")
        x = self.bert(input_ids, attention_mask=attention_mask)[0]
        x = GlobalAveragePooling1D()(x)
        x = Dense(512, activation=self.activation, kernel_regularizer=l2(0.001))(x)
        out = Dropout(0.2)(x)
        x = Dense(128, activation=self.activation, kernel_regularizer=l2(0.001))(x)
        out = Dropout(0.2)(x)

        model = Model(inputs=[input_ids, attention_mask], outputs=out)
        model.layers[2].trainable = False
        return model
        
        
        
        
