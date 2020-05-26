import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import itertools
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
import time
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


def run_clf(clf_name, X_train, y_train, X_test, y_test): # input should only be which classifier we want and data
    '''This function takes in training/testing data and a classifier name request, performs a cross-validated
    hyperparameter grid search and returns a prediction vector for testing data. It will be easy to add cost
    sensitivity to SVM's and any new classifiers at a later time.'''
    if clf_name == 'rbf':
        param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 'gamma': [10,1,0.1,0.01,0.001, 0.0001, 0.00001]}
        ]
        clf_gs = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
        clf_gs.fit(X_train, y_train)
        clf_rbf = SVC(C=clf_gs.best_params_.get('C'), gamma=clf_gs.best_params_.get('gamma'), kernel = 'rbf', probability=True) #class_weight='balanced'
        clf_rbf.fit(X_train, y_train)
        probs_clf = clf_rbf.predict_proba(X_test)[:, 1]
        y_pred = clf_rbf.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('AUC:', roc_auc_score(y_test, probs_clf))
        
    if clf_name == 'lin':
        param_grid = [
        {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]}
        ]
        clf_gs = GridSearchCV(LinearSVC(), param_grid, cv=5)
        clf_gs.fit(X_train, y_train)
        clf_lin = LinearSVC(C=clf_gs.best_params_.get('C'))
        clf_lin.fit(X_train, y_train)
        probs_clf = clf_lin.decision_function(X_test)
        y_pred = clf_lin.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('AUC:', roc_auc_score(y_test, probs_clf))
        
    if clf_name == 'rf':
        rfc = RandomForestClassifier(n_jobs = -1, max_features = 'sqrt', n_estimators=32, oob_score = True)
        param_grid = {
            'n_estimators': [50, 100, 200, 500, 1000],
            'max_features': ['sqrt', 'log2']
        }
        clf_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv = 5)
        clf_rfc.fit(X_train, y_train)
        clf_rf_1 = RandomForestClassifier(n_jobs=-1, n_estimators=clf_rfc.best_params_.get('n_estimators'), max_features=clf_rfc.best_params_.get('max_features'), oob_score = True)
        clf_rf_1.fit(X_train, y_train)
        probs_clf = clf_rfc.predict_proba(X_test)[:, 1]
        y_pred = clf_rfc.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('AUC:', roc_auc_score(y_test, probs_clf))
        
    if clf_name == 'log':
        param_grid = [
            {'C': [0.001, 0.1, 1, 10, 1000], 'penalty':['l1','l2'] }
        ]
        clf_gs = GridSearchCV(LogisticRegression(), param_grid, cv = 5)
        clf_gs.fit(X_train, y_train)
        clf_log = LogisticRegression(C = clf_gs.best_params_.get('C'), penalty =clf_gs.best_params_.get('penalty'), tol=0.01)
        clf_log.fit(X_train, y_train)
        probs_clf = clf_log.predict_proba(X_test)[: ,1]
        y_pred = clf_log.predict(X_test)
        print('Accuracy:', accuracy_score(y_test, y_pred))
        print('AUC:', roc_auc_score(y_test, probs_clf))
        
