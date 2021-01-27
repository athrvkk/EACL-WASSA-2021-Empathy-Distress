# File of the class Utils, containing various helper functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np
import re
import unicodedata
from pycontractions import Contractions


class Utils():
    """" Class containing various helper functions """   
    
    
    # -------------------------------------------- Class Constructor --------------------------------------------
    
    def __init__(self, contractions_model_path="/home/eastwind/word-embeddings/word2vec/GoogleNews-vectors-negative300.bin"):
        """ Class Constructor
        @param contractions_model_path (str): model to be loaded for contractions expansion.
        """
        self.cont = Contractions(contractions_model_path)
        self.cont.load_models()
    
    
    
    
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
    
    
    
    
    # -------------------------------------------- Function to expand contractions --------------------------------------------
    
    def expand_contractions(self, text):
        """ Function to expand contractions
        @param text (str): input text to euxpand contractions.
        return text (str): Contraction expanded text.
        """
        text = list(cont.expand_texts([text], precise=True))[0]
        return text
    
    
    
    
    # -------------------------------------------- Function to normalize input text --------------------------------------------
    
    def normalize_text(self, text):
        """ Function to normalzie text inputs.
        @param text (str): Input text.
        """
        # Adding space for all puntuation marks
        text = re.sub(r"([\w/'+$\s-]+|[^\w/'+$\s-]+)\s*", r"\1 ", text)

        # Remove accented words
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        # Remove long spaces
        text = re.sub(r'^\s*|\s\s*', ' ', text).strip()

        tokenized_text = text.split()
        abbr_dict = self.get_dict("../text-normalization-dictionaries/social-media-abbreviations.csv")
        for i in range(len(tokenized_text)):
            x = re.sub(r'[^\w\s]', '', tokenized_text[i]).lower()
            
            # Expand acronyms
            if x in abbr_dict.keys():
                tokenized_text[i] = abbr_dict[x]

            # Expand contracitons
            tokenized_text[i] = self.expand_contractions(tokenized_text[i])    

        # Combine words into sentence.    
        text = ""
        for word in tokenized_text:
            text = text + " " + word
        
        return text
        

        
      
    # -------------------------------------------- Function to Normalize corpus --------------------------------------------
        
    def normalize_corpus(self, df, column_name):
        """ Function to normalize corpus.
        @param corpus (list): corpus list.
        @param column_name (str): name of column to normalize.
        """
        df[column_name] = df[column_name].apply(lambda text: self.normalize_text(text))
        return df
        
        
        
