# File of the class Utils, containing various helper functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences




class Utils():
    """" Class containing various helper functions """   
    
    
    
    def __init__(self):
        """ Class Constructor """
        self.tokenizer = Tokenizer(oov_token='[oov]')




    # -------------------------------------------- Function to read data --------------------------------------------
    
    def read_data(self, path, columns=[]):
        """ Function to read data
        @param path (str): path to the data to be read.
        @columns (list): List of columns to be read.
        return data (pandas.DataFrame): a dataframe of the data read.
        """
        delimiter = path.split(".")[-1]

        if delimiter == "csv":
            if not columns:
                data = pd.read_csv(path, sep=",")
                return data
            else:
                data = pd.read_csv(path, sep=",", usecols=columns)
                return data

        elif delimiter == "tsv":
            if not columns:
                data = pd.read_csv(path, sep="\t")
                return data
            else:
                data = pd.read_csv(path, sep="\t", usecols=columns)
                return data

        else:
            return None
            
   
   
            
    # -------------------------------------------- Function read dictionary --------------------------------------------
    
    def get_dict(self, path):
        """ Function to read a file into dictionary.
        @param path (str): path to file.
        return dict: created dictionary.
        """
        data = pd.read_csv(path)
        return dict(zip(data.iloc[:,0].tolist(), data.iloc[:,1].tolist())) 



       
    # -------------------------------------------- Function to tokenize and pad input text --------------------------------------------    

    def prepare_input(self, corpus, maxlen=100, padding_type='post', truncating_type='post', mode="train"):
        """ Function to prepare text for model input (tokenize and pad).
        @param corpus (list): the corpus to prepare.
        @param maxlen (int): max allowed length of input texts.
        @param padding_type (str): padding type (post or pre).
        @param truncating_type (str): truncating type (post or pre).
        @mode (str): specify train or test mode.
        """
        if mode == "train":
            self.tokenizer.fit_on_texts(corpus)

        corpus = self.tokenizer.texts_to_sequences(corpus)
        corpus = np.asarray(pad_sequences(corpus, 
                                          padding=padding_type,
                                          truncating=truncating_type,
                                          maxlen=maxlen))
        return corpus, self.tokenizer 




    # -------------------------------------------- Function to read embeddings -------------------------------------------- 
    
    def get_embedding_matrix(self, path, vocab, top=50000):
        """ Function to get the word embedding matrix.
        @param path (str): path to the word embeddings file.
        @param vocab (list): list of corpus vocab.
        @return embedding_matrix (np.array): embedding matrix.
        """
        if path.split(".")[-1] == "bin":
            embedding_model = KeyedVectors.load_word2vec_format(path, binary=True, limit=top)
        else:
            embedding_model = KeyedVectors.load_word2vec_format(path, binary=False, limit=top)
        
        embeddin_model_vocab =[word for word in embedding_model.vocab.keys()]
        final_vocab = list(set(vocab) | set(embeddin_model_vocab))      
        
        embedding_matrix = np.zeros((len(final_vocab), 300))
        cnt = 0
        for index in range(len(final_vocab)):
            if final_vocab[index] in embeddin_model_vocab:
                vec = embedding_model.wv[final_vocab[index]]
                if vec is not None:
                    embedding_matrix[index] = vec
            else:
                cnt = cnt + 1
                continue       
        print("zero embeddings: ", cnt)
        return embedding_matrix
