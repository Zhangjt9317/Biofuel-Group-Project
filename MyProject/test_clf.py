import clfdata
from clfdata import *
import numpy as np

X_train, y_train, X_test, y_test = data_clean()
X=X_train.iloc[0:1,:]
y_pred = 'Esters'

def test_data_clean():
    # four results of data sets should be provided
    assert len(data_clean()) == 4, "clean_data() breaks"
    assert type(data_clean()) == tuple, "clean_data() breaks"
    return 

def test_train_knn():
    k = 5
    knn = train_knn(k, X_train, y_train)
    # knn should be a classifier
    type(knn) == KNeighborsClassifier, "train_knn() breaks"
    return 

def test_test_knn():
    k = 5
    # accuarcy should be possitive
    assert test_knn() > 0, "test_knn() breaks"
    return
    

def test_predict_family_knn():
    # prediction should be an array of family
    assert type(predict_family_knn(X)) == np.ndarray, "predict_family_knn() breaks"
    return

def test_plot_knn():
    # this fuction should provide a figure
    assert type(plot_knn(y_pred)) == plt.Figure, "plot_knn() breaks"
    return 

def test_train_lda():
    # lda should be a classifier
    lda = train_lda(X_train, y_train)
    type(lda) == LinearDiscriminantAnalysis, "train_lda() breaks"
    return 

def test_test_lda():
    # accuarcy should be possitive
    assert test_lda() > 0, "test_lda() breaks"
    return
    

def test_predict_family_lda():
    # prediction should be an array of family
    assert type(predict_family_lda(X)) == np.ndarray, "predict_family_lda() breaks"
    return

def test_plot_lda():
    # this fuction should provide a figure
    assert type(plot_lda(y_pred)) == plt.Figure, "plot_lda() breaks"
    return

def test_train_svm():
    # svm should be a classifier
    svm = train_svm(X_train, y_train)
    type(svm) == LinearSVC, "train_svm() breaks"
    return 

def test_test_svm():
    # accuarcy should be possitive
    assert test_svm() > 0, "test_svm() breaks"
    return
    
def test_predict_family_svm():
    # prediction should be an array of family
    assert type(predict_family_svm(X)) == np.ndarray, "predict_family_svm() breaks"
    return

def test_plot_svm():
    # this fuction should provide a figure
    assert type(plot_svm(y_pred)) == plt.Figure, "plot_svm() breaks"
    return 