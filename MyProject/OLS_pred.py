from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np

def OLS_pred(family, prop):
    """
    This function is used to predict properties according to Ordinary Least Squares(linear model). 
    """
    data = Database()   ###load, select, clear NaN data
    data_f = data[data.Family == family]
    df = data_f[np.isfinite(data_f[prop])]
    test_size = 0.1  ###define the test_size
    train, test = train_test_split(df, test_size=test_size)  ###split data
    OLS = linear_model.LinearRegression()   ###build model
    train_X = train[train.columns[4:]]   ###select functional groups
    OLS.fit(train_X, train[prop])    ###train model
    return OLS, train, test   ###return model, train, test data to plot

def OLS_plot(family, prop, iteration, fraction):
    """
    This function is used to make plots according to OLS model.
    """
    model, train, test = OLS_pred(family, prop)
    plot(train, test, iteration, fraction, model, prop, family)  ###make plots