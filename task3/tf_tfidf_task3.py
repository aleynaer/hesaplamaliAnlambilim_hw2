# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 03:57:39 2022

@author: Asus

tf on news data
"""
import os
os.chdir(r"C:\Users\Asus\Desktop\Anlambilim_odev2")
os.getcwd()
#%%
# load dataset
addr = r"C:\Users\Asus\Desktop\Anlambilim_odev2\task3\texts\dataset\*txt"

corpus = [] 

import glob 
import errno 
 
files = glob.glob(addr) 
for name in files: 
    try: 
        with open(name) as f: 
            raw_text = f.read()
            corpus.append(raw_text)
            f.close()
    except IOError as exc: 
        if exc.errno != errno.EISDIR: 
            raise 
print(corpus)
#%%
for i in corpus:
    i = " ".join(i.split())

import pandas as pd
data = pd.DataFrame(corpus, columns = ["docs"])
data

#%%
query_addr =  r"C:\Users\Asus\Desktop\Anlambilim_odev2\task3\texts\query\A11.txt"

with open(query_addr) as f: 
        query_text = f.read()
        f.close()
        
query_text
#%% add query to data
query_text = {"docs":query_text}
data = data.append(query_text, ignore_index=True)

#%% tf hesaplamak için vocab oluştur - 3 karakter gram
vocab = set()
from nltk import ngrams

for i in range(len(data)):
    desc = data["docs"].values[i]
    desc = desc.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
    desc = ' '.join(desc.split()) #a quick way of removing excess whitespace
    punc = '''!()-[]{};:""''\<>.@#$%^&*_?,''' # ?/
    
    for element in desc:
        if element in punc:
            desc = desc.replace(element,"")
            
    data["docs"].values[i] = desc
    
    a = ["".join(k1) for k1 in list(ngrams(desc,n=3))]
    for j in a:
      #print(j)
      vocab.add(j)
      
    
#%%
vocab = list(vocab)
data = data.reindex(columns=data.columns.tolist() + vocab) 
data.head()
#%% tf hesapla
for i in range(len(data)):
    desc = data["docs"].values[i]
    counter = 0
    for j in vocab:
        if j in desc:
            counter += 1
        data[j].values[i] = counter
        counter = 0
    print(i)
#%%  compute cos sim query with docs
import numpy as np

def cos_sim(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_1 = np.linalg.norm(vector1)
    norm_2 = np.linalg.norm(vector2)
    return dot_product / (norm_1 * norm_2)

#%%
query_vector = data.iloc[:,1:].values[10]
print("dokümanları tf ile vektörize edince; ")
for i in range(len(data)-1):
    base_vector = data.iloc[:,1:].values[i]
    similarity = cos_sim(base_vector, query_vector)
    print("doküman {0} ile sorgunun benzerliği: {1}".format(i+1,similarity))

#%% tf-idf tablosu oluştur, idf hesapla
import math
data_copy = data.copy()
doc_num = len(data_copy)

cols = data_copy.iloc[:,1:].columns

for col in cols:
    count = 0 # kelimenin dokümanda geçme sayısını tutar
    for i in range(len(data_copy)):
        tf = data_copy[col].values[i]
        if(tf > 0): # tf>0 o dokümanda geçmiştir
            count += 1 
    idf = doc_num/count
    idf = math.log(idf) # idf hesaplanır
    #print(idf)
    for x in range(len(data_copy)): # tfidf hesaplayıp bastır
        tf = data_copy[col].values[x]
        tf = tf + 0.5
        tf = math.log(tf)
        #print(tf)
        tfidf = tf*idf
        #print(tfidf)
        data_copy[col].values[x] = tfidf
#%%
query_vector = data_copy.iloc[:,1:].values[10]
print("dokümanları tfidf ile vektörize edince; ")
for i in range(len(data)-1):
    base_vector = data_copy.iloc[:,1:].values[i]
    similarity = cos_sim(base_vector, query_vector)
    print("doküman {0} ile sorgunun benzerliği: {1}".format(i+1,similarity))
    
#%% save the tf and tfidf data
data.to_csv("task3tf.csv",index=False)
data_copy.to_csv("task3tfidf.csv",index=False)
