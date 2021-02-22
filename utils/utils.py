# File of the class Utils, containing various helper functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer



class Utils():
    """" Class containing various helper functions """   
    
    
    # -------------------------------------------- Class Constructor --------------------------------------------
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
    
    def get_dict(self, path, key_column, value_column):
        """ Function to read a file into dictionary.
        @param path (str): path to file.
        return dict: created dictionary.
        """
        data = pd.read_csv(path)
        return dict(zip(data[key_column].values.tolist(), data[value_column].values.tolist()))



       
    # -------------------------------------------- Function to tokenize and pad input text --------------------------------------------    

    def tokenize_and_pad(self, corpus, maxlen=100, padding_type='post', truncating_type='post', mode="train"):
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
        
        
        
     
     # -------------------------------------------- Function to create age bins --------------------------------------------
    
    def categorize_age(self, age):
        if age >0 and age <=25:
            return 0
        elif age >25 and age <=40:
            return 1
        elif age >40 and age <=60:
            return 2
        elif age >60:
            return 3
            
            
            
            
    # -------------------------------------------- Function to create age bins --------------------------------------------
    
    def categorize_income(self, income):
        if income >0 and income <=25000:
            return 0
        elif income >25000 and income <=75000:
            return 1
        elif income >75000:
            return 2
      
            
            
            
                       
    # ----------------------------------- Function to calculate weight of each word -----------------------------------

    def get_word_weights(self, essay, feature):
        ''' Function to calculate weight of each word according to the specified feature
        @param essay (pd.Series) list of essays.
        @param feature (pd.Series) The target variable (gold_empthy_bin or gold_distres_bin).
        @return weight (dict) The dictionary contribution of each word to the feature passed.
        '''
        dictionary1 = {}   #to count the number of times a word contributes to feature=1
        dictionary0 = {}   #to count the number of times a word contributes to feature=0
        words=[] 
        for i in range(0,len(essay)):   #loop to find both counts(feature=1, feature=0) for each word
            words = essay[i].split()
            for word in words:
                if word.isalpha():
                    if word in dictionary1.keys():
                        if feature[i] == 1:
                            dictionary1[word] = dictionary1[word] + 1
                        else :
                            dictionary0[word] = dictionary0[word] + 1
                    else:
                        if feature[i] == 1:
                            dictionary1[word] = 1
                            dictionary0[word] = 0
                        else:
                            dictionary0[word] = 1
                            dictionary1[word] = 0
                    
        weight = {}  #to store weight of each word
        for i in dictionary0:
            weight[i]=((dictionary1[i]-dictionary0[i])/(dictionary1[i]+dictionary0[i]))
    
        return weight




    # ----------------------------------- Function to calculate score for each Essay -----------------------------------

    def get_essay_empathy_distress_scores(self, essay, word_weights, transform='original'):
        ''' Function to calculate score for each Essay.
        @param essay (list) list of essays.
        @param word_weights (dict) Tthe dictionary obtained from the wordWeight function
        @return essay_weight (list): list of weights for each essay.
        '''
        essay_weight = []
        for i in range(0, len(essay)):
            weight = 0
            for word in essay[i].split():
                if word in word_weights.keys():
                    weight = weight + float(word_weights[word])
            essay_weight.append(weight)
        if transform == "original":
            return np.reshape(essay_weight, (len(essay_weight), 1))
        elif transform == "tan-inverse":    
            np.arctan(np.reshape(essay_weight, (len(essay_weight), 1)))




    # ----------------------------------- Function to get word emotion and vad scores -----------------------------------
            
    def get_word_scores(self, sentiment='anger'):
        path = "/content/gdrive/My Drive/WASSA-2021-Shared-Task/resources/NRC-resources/"+sentiment+"-scores.txt"
        word_scores = {}
        with open(path) as f:
            data = f.readlines()
            for row in data:
                row = row.split("\t")
                row[1] = row[1].split("\n")
                word_scores[row[0]] = row[1][0]
        f.close()
        return word_scores
        
    
    
    
    # -------------------------------------------- Function to get essay emotion and vad scores --------------------------------------------
    
    def get_essay_nrc_scores(self, essay, nrc_features, normalize=False):
        word_scores_list = []
        for element in nrc_features:
            word_scores = self.get_word_scores(element)
            word_scores_list.append(word_scores)
        essay_scores = np.zeros((len(essay), len(word_scores_list)))
        
        for i in range(len(essay)):        
            for j in range(len(word_scores_list)):
                score = 0
                cnt = 0
                for word in essay[i].split():
                    if word in word_scores_list[j].keys():
                        cnt = cnt = 1
                        score = score + float(word_scores_list[j].get(word))
                essay_scores[i][j] = score
        
        return essay_scores
                
        
