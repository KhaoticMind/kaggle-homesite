# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from helper import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import normalize
import pandas as pd
from ipyparallel import Client
from sklearn.preprocessing import LabelEncoder
import numpy as np
from xgboost import XGBClassifier


class HomeSiteDataTransform1():
    def fit(self, X, y=None):
        pass

    def transform(self, X, y=None):
        X['Date'] = pd.to_datetime(pd.Series(X['Original_Quote_Date']))
        X['Year'] = X['Date'].apply(lambda x: int(str(x)[:4]))
        X['Month'] = X['Date'].apply(lambda x: int(str(x)[5:7]))
        X['weekday'] = X['Date'].dt.dayofweek

        X.drop(['QuoteConversion_Flag', 'QuoteNumber', 'Original_Quote_Date', 'Date'], axis=1, inplace=True)

        for f in X.columns:
            if X[f].dtype == 'object':
                print(f)
                lbl = LabelEncoder()
                lbl.fit(list(X[f].values))
                X[f] = lbl.transform(list(X[f].values))

        X.fillna(0, inplace=True)
        return X.values

train, test = data_load()
y_train = train.QuoteConversion_Flag.values
X = HomeSiteDataTransform1().transform(train)
X = X[: 1000]
y_train = y_train[:1000]

del test
del train

classifiers = []

rf_clf = RandomForestClassifier(n_estimators=100)
params_rf = {'criterion': ['gini', 'entropy'],
             'oob_score': [True, False],
             'max_features': ['sqrt', 'log2']}
classifiers.append(('rf', rf_clf, params_rf, None))

lsvc_clf = LinearSVC()
params_lsvc = {'C': np.logspace(-3, 2, num=5),
               'tol': np.logspace(-6, -2, num=5)}
#classifiers.append(('lsvc', lsvc_clf, params_lsvc, None))

knn_clf = KNeighborsClassifier()
params_knn = {'n_neighbors': np.linspace(5, 50, 5),
              'p': np.linspace(1, 5, 5),
              'algorithm': ['ball_tree', 'kd_tree', 'brute']}
classifiers.append(('knn', knn_clf, params_knn, None))

svm_clf = SVC()
params_svm = {'C': np.logspace(-3, 2, num=5),
              'gamma': np.logspace(-6, -2, num=5),
              'tol': np.logspace(-6, -2, num=5),
              'shrinking': [True, False]}
#classifiers.append(('svm', svm_clf, params_svm, None))

xgb_clf = XGBClassifier(nthread=1)
params_xgb = {'max_depth': np.linspace(3, 15, 5),
              'learning_rate': np.linspace(0.001, 0.1, 5),
              'subsample': np.linspace(0.1, 0.9, 5),
              'colsample_bytree': np.linspace(0.1, 0.9, 5)}
fit_params_xgb = {'eval_metric': 'auc'}
classifiers.append(('xgb', xgb_clf, params_xgb, fit_params_xgb))

client = Client()
lbview = client.load_balanced_view()

Xnorm = normalize(X)
dados = [('dado', X), ('dado_norm', Xnorm)]
dados_label = ['dado', 'dado_norm']

for label, X in dados:
    applyPerHost(client, persistData, X, label)

applyPerHost(client, persistData, y_train, 'y_values')

tasks = modelSearch(lbview, dados_label, 'y_values',
                    X.shape[0], classifiers, metric='auc')

results = getResults(tasks)

res = results.groupby(['label_clf', 'label_dados', 'params'], as_index=False).\
                    mean()
res = res.groupby(['label_clf', 'label_dados'], as_index=False).max()
