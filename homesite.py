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
from sklearn.feature_selection import SelectFpr, f_classif, chi2

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

client = Client()
lbview = client.load_balanced_view()

train, test = data_load()
n_elements = np.floor(train.shape[0] * 0.05)
train = train.iloc[:n_elements, ]

y_train = train.QuoteConversion_Flag.values

dados_label = []
dados = []

fpr_f_classif = SelectFpr(f_classif)
fpr_chi2 = SelectFpr(chi2)

X = HomeSiteDataTransform1().transform(train)
dados_label.append('dado')
dados.append(X)

Xnorm = normalize(X)
dados_label.append('dado_norm')
dados.append(Xnorm)

dados_label.append('dado_fpr_f_classif')
dados.append(fpr_f_classif.fit_transform(X, y_train))

dados_label.append('dado_fpr_chi2')
dados.append(fpr_chi2.fit_transform(X, y_train))

dados_label.append('dado_norm_fpr_f_classif')
dados.append(fpr_f_classif.fit_transform(Xnorm, y_train))

dados_label.append('dado_norm_fpr_chi2')
dados.append(fpr_chi2.fit_transform(Xnorm, y_train))

# Store  the data
todos_dados = list(zip(dados_label, dados))

for label, X in todos_dados:
    applyPerHost(client, persistData, X, label)

applyPerHost(client, persistData, y_train, 'y_values')

del test
del train
del dados
del todos_dados


classifiers = []

rf_clf = RandomForestClassifier(n_estimators=100)
params_rf = {'criterion': ['gini', 'entropy'],
             'oob_score': [True, False],
             'max_features': ['sqrt', 'log2']}
classifiers.append(('rf', rf_clf, params_rf, None))

lsvc_clf = LinearSVC()
params_lsvc = {'C': np.logspace(-3, 2, num=5),
               'tol': np.logspace(-6, -2, num=5)}
# classifiers.append(('lsvc', lsvc_clf, params_lsvc, None))

knn_clf = KNeighborsClassifier()
params_knn = {'n_neighbors': np.linspace(5, 50, 5),
              'p': np.linspace(1, 5, 5),
              'algorithm': ['ball_tree', 'kd_tree', 'brute']}
# classifiers.append(('knn', knn_clf, params_knn, None))

svm_clf = SVC()
params_svm = {'C': np.logspace(-3, 2, num=5),
              'gamma': np.logspace(-6, -2, num=5),
              'tol': np.logspace(-6, -2, num=5),
              'shrinking': [True, False]}
# classifiers.append(('svm', svm_clf, params_svm, None))

xgb_clf = XGBClassifier(nthread=1)
params_xgb = {'max_depth': np.linspace(3, 15, 5),
              'learning_rate': np.linspace(0.001, 0.1, 5),
              'subsample': np.linspace(0.1, 0.9, 5),
              'colsample_bytree': np.linspace(0.1, 0.9, 5)}
fit_params_xgb = {'eval_metric': 'auc'}
classifiers.append(('xgb', xgb_clf, params_xgb, fit_params_xgb))

tasks = modelSearch(lbview, dados_label, 'y_values',
                    X.shape[0], classifiers, metric='auc')

results = getResults(tasks)

res = results.groupby(['label_clf', 'label_dados', 'params'],
                      as_index=False).mean()
res_idxs = res.groupby(['label_clf', 'label_dados'],
                       as_index=False)['test_score'].idxmax()
res = res.iloc[res_idxs]

del X
del Xnorm