import matplotlib.pyplot as plt 
import pandas as pd



def scatterr(x, xlabel):
    df = pd.read_csv (r'dataset\my_data.csv')
    fig, (ax, ax1) = plt.subplots(2)
    ax.scatter(x,df['empathy'])
    ax.set(xlabel = xlabel)
    ax.set(ylabel = 'empathy')
    ax1.scatter(x,df['distress'])  
    ax1.set(xlabel = xlabel)
    ax1.set(ylabel = 'distress')



def scatterr_bin(x, xlabel):
    df = pd.read_csv (r'dataset\my_data.csv')
    fig, (ax, ax1) = plt.subplots(2)
    ax.scatter(x,df['empathy_bin'])
    ax.set(xlabel = xlabel)
    ax.set(ylabel = 'empathy_bin')
    ax1.scatter(x,df['distress_bin'])  
    ax1.set(xlabel = xlabel)
    ax1.set(ylabel = 'distress_bin')
