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
addr_f = "clean_fake.txt"

with open(addr_f) as f: 
    fake_headers = f.read().splitlines()
    f.close()

addr_r = "clean_real.txt"

with open(addr_r) as f: 
    real_headers = f.read().splitlines()
    f.close()
#%%
import pandas as pd
fake = pd.DataFrame(data = fake_headers, columns = ["header"])
fake["isFake"] = 1
fake.head()

real = pd.DataFrame(data = real_headers, columns = ["header"])
real["isFake"] = 0
real.head()

data = pd.concat([fake,real], axis = 0)
data
#%% tf hesaplamak için vocab oluştur - 3 karakter gram
vocab = set()
from nltk import ngrams

for i in range(len(data)):
    desc = data["header"].values[i]
    desc = desc.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
    desc = ' '.join(desc.split()) #a quick way of removing excess whitespace
    punc = '''!()-[]{};:""''\<>.@#$%^&*_?,''' # ?/
    
    for element in desc:
        if element in punc:
            desc = desc.replace(element,"")

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
    desc = data["header"].values[i]
    counter = 0
    for j in vocab:
        if j in desc:
            counter += 1
        data[j].values[i] = counter
        counter = 0
    print(i)
#%%  evaluate a knn model using k-fold cross-validation
# vectorized with fasttext
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()

from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

# create dataset
X = data.iloc[:,2:].values
y = data["isFake"].values
# prepare the cross-validation procedure
cv = KFold(n_splits=5, random_state=1, shuffle=True)
# create model
# evaluate model
scores = cross_val_score(knn_clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('TF Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))

#%% tf-idf tablosu oluştur, idf hesapla
import math
data_copy = data.copy()
doc_num = len(data_copy)

cols = data.iloc[:,2:].columns

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
X1 = data_copy.iloc[:,2:].values
y1 = data_copy["isFake"].values
scores1 = cross_val_score(knn_clf, X1, y1, scoring='accuracy', cv=cv, n_jobs=-1)
print('TFIDF Accuracy: %.3f (%.3f)' % (mean(scores1), std(scores1)))
#%% save the tf and tfidf data
data.to_csv("task2tf.csv",index=False)
data_copy.to_csv("task2tfidf.csv",index=False)
