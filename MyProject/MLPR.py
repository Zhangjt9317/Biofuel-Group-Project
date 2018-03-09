import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

import database
import df_prediction


def MLPR(family, prop, test_size):
    """This function is used to predict properties by using the Multiple Layers Perception Regression model."""
    # Input data and define the parameters
    df = database.Database()
    train, test = df_prediction.df_prediction(family, prop, test_size)
    x_train = train[train.columns[4:]]
    y_train = train[prop]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(x_train)
    np.set_printoptions(precision=4) # summarize transformed data for x,, and also set up the descimal place of the value
        
    mlpr = MLPRegressor(hidden_layer_sizes=(1000,),activation='identity', solver='sgd', learning_rate='adaptive', max_iter=4000, verbose=False)
    mlpr.fit(x_train, y_train)
    return mlpr, train, test
