import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection 
from sklearn.model_selection import train_test_split
from database import Database
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.svm import LinearSVC


def data_clean():
    """split data and generate train & test subset"""
    df = Database()
    train, test = train_test_split(df, test_size=0.1)
    a = train.loc[:,'[H]': '[cX3H0](:*)(:*):*']
    X_train = a.mask(a>0, 1)
    y_train = train['Family']

    b = test.loc[:,'[H]': '[cX3H0](:*)(:*):*']
    X_test = b.mask(b>0, 1)
    y_test = test['Family']
    return X_train, y_train, X_test, y_test


def train_knn(k, X_train, y_train):
    """use knn method to train data"""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def test_knn(k, X_train, y_train, X_test, y_test):
    """plot train and test classification result"""
    knn = train_knn(k, X_train, y_train)
    y_pred = knn.predict(X_test)
    train_pred = knn.predict(X_train)
    print('Accuracy = ', metrics.accuracy_score(y_test, y_pred))
    plt.figure(figsize=(10,8))
    plt.scatter(y_test, y_pred, marker='s', s=100, label='test')
    plt.scatter(y_train, train_pred, marker='d', s=60,c='orange', label='train')
    plt.plot([0,7], [0,7], color='k')
    plt.xticks(rotation='40')
    plt.xlabel('Actual Family', fontsize=15)
    plt.ylabel('Predicted Family', fontsize=15)
    plt.title('Knn Classification',fontsize=20)
    plt.legend()
    return

def predict_family_knn(k, X_train, y_train, X):
    """predit family of import molecule X"""
    knn = train_knn(k, X_train, y_train)
    y_pred = knn.predict(X)
    return y_pred
