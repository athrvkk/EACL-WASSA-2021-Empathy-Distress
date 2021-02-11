import pandas as pd 
import numpy as np 
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score,mean_squared_error, accuracy_score
from scipy.stats import pointbiserialr
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

# ----------------------------------- Function to calculate weight of each word --------------------------------------------

def wordWeight(essay,feature):
    
    '''@param essay(pd.Series) is the "essay" column of the dataset
    feature(pd.Series) is either empathy_bin or distress_bin column of dataset
    return weight(dict) is the contribution of each word to the feature passed'''
    
    dictionary1 = {}   #to count the number of times a word contributes to feature=1
    dictionary0 = {}   #to count the number of times a word contributes to feature=0
    words=[] 
    for i in range(0,len(essay)):   #loop to find both counts(feature=1, feature=0) for each word
        words = essay[i].split()
        words = set(words)  
        words = (list(words)) 
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


# ------------------------------------- Function to calculate score for each Essay --------------------------------------------
def essayScore(data,weight):
    
    '''@param data(pd.DataFrame) is the complete dataset
       weight(dict) is the dictionary obtained from the wordWeight function
       return data is the complete dataset with an added column of "weight" for each essay'''
    
    zeroArray=np.zeros(shape=(len(df["essay"]),1))
    data["weight"]=pd.DataFrame(zeroArray)
    for i in range(0, len(df["essay"])):
        words = data["essay"][i].split()
        for word in words:
            if word.isalpha():
              if word in weight.keys():
                data["weight"][i]=data["weight"][i]+weight[word]
    return data

# ----------------------------- Function to calculate correlation between two features -----------------------------------------


def correlation(feature,target,flag):
    
    '''@param feature(pd.Series)
    target.series(pd.Series)
    flag(int)
    return x (float) which is the correlation between feature and target'''
    
    if(flag==0):
        x=target.corr(feature)
        return x
    if(flag==1):
        x=target.corr(np.arctan(feature))
        return x
    
    return "Enter valid flag"
    