from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np

def OLS_pred(family, prop, test_size):
    """
    This function is used to predict properties according to Ordinary Least Squares(linear model). 
    """
    train, test = df_prediction(family, prop, test_size)  ###create data for train and test
    OLS = linear_model.LinearRegression()   ###build model
    train_X = train[train.columns[4:]]   ###select functional groups
    OLS.fit(train_X, train[prop])    ###train model
    return OLS, train, test   ###return model, train, test data to plot

def OLS_plot(family, prop, iteration, fraction, test_size):
    """
    This function is used to make plots according to OLS model.
    """
    model, train, test = OLS_pred(family, prop, test_size)
    plot(train, test, iteration, fraction, model, prop, family)  ###make plots
    return