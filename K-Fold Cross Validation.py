#!/usr/bin/env python
# coding: utf-8

# Baginda - 130-
# coding ini saya save as dari file Tugas 1 tentang Genetic Algorithm, sehingga mohon maklum apabila "Tanggal Dibuat" yang tertera pada file ini adalah xxx
# coding ini saya ambil dan modifikasi sedikit dari website https://medium.com/datadriveninvestor/k-fold-cross-validation-6b8518070833
# apabila ada ketidaknyamanan dan ketidakpuasan dalam pemeriksaan coding ini, saya mohon maaf yang sebesar-besarnya
# saya akan sangat menerima apabila tugas ini dinilai seadanya
# terima kasih kepada dosen yang sudah memberikan ilmunya satu semester ini. mohon maaf, ternyata saya masih bodoh pada mata pelajaran ini
# Salam

# In[288]:


import pandas
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import numpy as np


# In[294]:


# membaca dataset	

dataset = pandas.read_csv('Diabetes.csv')


# In[301]:


# pre-processing

X = dataset.iloc[:, [0, 7]]
y = dataset.iloc[:, 8]


# In[304]:


# minmax scalling

scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X)

# In[310]:

# k-fold cross validation

scores = []
best_svr = SVR(kernel='rbf')
cv = KFold(n_splits=5, random_state=42, shuffle=False)
for train_index, test_index in cv.split(X):
    print("Train Index: ", train_index, "\n")
    print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
    best_svr.fit(X_train, y_train)
    scores.append(best_svr.score(X_test, y_test))
	


# In[317]:

print(np.mean(scores))
cross_val_score(best_svr, X, y, cv=10)
cross_val_predict(best_svr, X, y, cv=10)