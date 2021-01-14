import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv (r'dataset\my_data.csv')

def text_preproc(x):
    x = x.lower()
    x = x.encode('ascii', 'ignore').decode()
    x = re.sub(r'https*\S+', ' ', x)
    x = re.sub(r'@\S+', ' ', x)
    x = re.sub(r'#\S+', ' ', x)
    x = re.sub(r'\'\w+', '', x)
    x = re.sub(r'\w*\d+\w*', '', x)
    x = re.sub(r'\s{2,}', ' ', x)
    return x

def vectorize(clm):
    vectorizer = TfidfVectorizer(stop_words='english')

    # Learn vocabulary from sentences. 
    vectorizer.fit(df['essay'])

    # Get vocabularies.
    vectorizer.vocabulary_

    vector_spaces = vectorizer.transform(clm)
    feature_names = vectorizer.get_feature_names()
    dense = vector_spaces.todense()
    denselist = dense.tolist()
    df1 = pd.DataFrame(denselist, columns=feature_names)
    return df1

def process_text():
    encode_label()
    df['essay'] = df.essay.apply(text_preproc)
    df1 = vectorize(df['essay'])
    return df1

def encode_label():
    le = LabelEncoder() 
    df['emotion_label']= le.fit_transform(df['emotion_label']) 