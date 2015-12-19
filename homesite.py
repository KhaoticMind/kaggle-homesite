# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
sys.path.append('C:\\Users\\ur57\\Documents\\Python Scripts\\helper\\')

from helper import modelSearch, getResults, persistData, applyPerHost
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import normalize
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

rf_clf = RandomForestClassifier(n_estimators=100)
params_rf = {'criterion': ['gini', 'entropy'],
             'oob_score': [True, False],
             'max_features':['sqrt', 'log2']}
classifiers.append(('rf', rf_clf, params_rf))

lsvc_clf = LinearSVC()
params_lsvc = {'C': np.logspace(-3, 2, num=5),
               'tol': np.logspace(-6, -2, num=5) }               
classifiers.append(('lsvc', lsvc_clf, params_lsvc))

knn_clf = KNeighborsClassifier()
params_knn = { 'n_neighbors' : np.linspace(5, 50, 5),
                              'p': np.linspace(1,5, 5),
                              'algorithm' : ['ball_tree', 'kd_tree', 'brute']}
classifiers.append(('knn', knn_clf, params_knn))

svm_clf = SVC()
params_svm = {'C': np.logspace(-3, 2, num=5),
               'gamma': np.logspace(-6, -2, num=5) ,
             'tol': np.logspace(-6, -2, num=5),
            'shrinking ': [True,False]
                }
#classifiers.append(('svm', svm_clf, params_svm))

client = Client()
lbview = client.load_balanced_view()

Xnorm = normalize(X)
dados = [('dado', X), ('dado_norm', Xnorm)]

for label, X in dados:    
    applyPerHost(client, persistData, X, label)

tasks = modelSearch(lbview, dados, y_train, classifiers)

results = getResults(tasks)

res = results.groupby(['label_clf', 'label_dados', 'params'], as_index=False).mean()
res = results.groupby(['label_clf', 'label_dados'], as_index=False).max()
