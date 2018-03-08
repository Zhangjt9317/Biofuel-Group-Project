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


def GRNN():
    """General Regression Neural Network"""
    df = database.Database()
    df = df[np.isfinite(df['Flash Point'])]
    xa = np.linspace(200, 550)
    x = df.loc[:,'[H]':'[cX3H0](:*)(:*):*']
    y = df['Flash Point']
    array_x = x.values
    array_y = y.values

    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(array_x)
    np.set_printoptions(precision=4) # summarize transformed data for x,, and also set up the descimal place of the value
    
    x_train, x_test, y_train, y_test = train_test_split(rescaledX, array_y, test_size=0.1, random_state=17)
    
    grnn = algorithms.GRNN(std=0.3,verbose=False,)
    return grnn
