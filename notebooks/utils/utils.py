# File of the class Utils, containing various helper functions
# File: utils.py
# Author: Atharva Kulkarni


import pandas as pd
import numpy as np


class Utils():
    """" Class containing various helper functions """   
    
    
    
    
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
        
        
        
