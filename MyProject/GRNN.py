import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from neupy import algorithms, layers, environment, estimators

import database
import df_prediction


def GRNN(family, prop, test_size):
    """This function is used to predict properties by using the General Regression Neural Network model."""
    train, test = df_prediction.df_prediction(family, prop, test_size)  ###create data for train and test
    x_train = train[train.columns[4:]]   ###select functional groups
    y_train = train[prop]

    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(x_train)
    np.set_printoptions(precision=4) # summarize transformed data for x,, and also set up the descimal place of the value
    
    grnn = algorithms.GRNN(std=0.3,verbose=False,)
    grnn.train(x_train, y_train)
    return grnn, train, test

def GRNN_plot(family, prop, iteration, fraction, test_size):
    """
    This function is used to make plots according to GRNN model.
    """
    model, train, test = GRNN(family, prop, test_size)
    plot(train, test, iteration, fraction, model, prop, family)  ###make plots
    return
