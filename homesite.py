# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 08:58:33 2015

@author: ur57
"""

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.preprocessing import LabelEncoder

import numpy as np
import pandas as pd


train = pd.read_csv('train.csv', na_values=-1)
test = pd.read_csv('test.csv', na_values=-1)

y_train = train.QuoteConversion_Flag.values
test_number = test['QuoteNumber']

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
train['Year']  = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test['Year']  = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek

train.drop(['QuoteConversion_Flag', 'QuoteNumber', 'Original_Quote_Date', 'Date'], axis=1, inplace=True)
test.drop(['QuoteNumber', 'Original_Quote_Date', 'Date'], axis=1, inplace=True)

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

#train = pd.get_dummies(train, sparse=True, columns=train.columns.drop('SalesField8')

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl = LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))            

n_folds=3
clfs= []
X = train.values

folds = KFold(X.shape[0], n_folds, shuffle=True)
for train_index, test_index in folds:
    clf = RandomForestClassifier(max_features='sqrt', criterion='entropy', n_estimators=100, oob_score=False)
    clf.fit(X[train_index], y_train[train_index])
    clfs.append(clf)
    
    clf = GradientBoostingClassifier()
    clf.fit(X[train_index], y_train[train_index])
    clfs.append(clf)

res = []
for clf in clfs:
    if hasattr(clf, "predict_proba"):
        y_pred = clf.predict_proba(test)[:,1]
    else:  # use decision function
        y_pred = clf.decision_function(test)
        y_pred = \
            (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min())
    
    res.append(y_pred)
    
final_res = np.mean(res, axis=0)

sub = pd.DataFrame()
sub['QuoteNumber'] = test_number
sub['QuoteConversion_Flag'] = final_res
sub.to_csv('my_sub.csv', index=False)

'''
params_rf = {'criterion': ['gini', 'entropy'],
             'oob_score': [True, False],
             'max_features':['sqrt', 'log2']}
clf = RandomizedSearchCV(RandomForestClassifier(n_estimators=200), param_distributions=params_rf, n_jobs=1,  scoring='roc_auc', verbose=4)
clf.fit(train, y_train)
y_pred = clf.predict_proba(test)
'''
