# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.append('C:\\Users\\ur57\\Documents\\Python Scripts\\helper\\')

from helper import modelSearch, getResults, persistData, applyPerHost
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import pandas as pd
from IPython.parallel import Client
from sklearn.preprocessing import LabelEncoder
import numpy as np

train = pd.read_csv('train.csv', na_values=-1, nrows=10000)
y_train = train.QuoteConversion_Flag.values

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train['Year']  = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

train.drop(['QuoteConversion_Flag', 'QuoteNumber', 'Original_Quote_Date', 'Date'], axis=1, inplace=True)

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values))
        train[f] = lbl.transform(list(train[f].values))

train.fillna(0, inplace=True)
X = train.values

classifiers = []

rf_clf = RandomForestClassifier()
params_rf = {'criterion': ['gini', 'entropy'],
             'oob_score': [True, False],
             'max_features':['sqrt', 'log2']}
classifiers.append(('rf', rf_clf, params_rf))

svc_clf = LinearSVC()
params_lsvc = {'C': np.logspace(-3, 2, num=10),
               'tol': np.logspace(-6, -2, num=10) }
classifiers.append(('lsvc', svc_clf, params_lsvc))


client = Client()
lbview = client.load_balanced_view()

dados = [('dado', X)]

for label, X in dados:    
    applyPerHost(client, persistData, X, label)

#tasks = modelSearch(lbview, dados, y_train, classifiers)

#results = getResults(tasks)

#results.groupby(['label_clf', 'params']).train_score.mean()
