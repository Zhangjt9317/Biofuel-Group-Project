from sklearn import linear_model
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

def PNR_pred(family, prop, test_size):
    """
    This function is used to predict properties according to Polynomial Regression(nonlinear model). 
    """
    train, test = df_prediction(family, prop, test_size)  ###create data for train and test
    PNR = make_pipeline(PolynomialFeatures(), Ridge())   ###build model
    train_X = train[train.columns[4:]]   ###select functional groups
    PNR.fit(train_X, train[prop])    ###train model
    return PNR, train, test   ###return model, train, test data to plot

def PNR_plot(family, prop, iteration, fraction, test_size):
    """
    This function is used to make plots according to OLS model.
    """
    model, train, test = PNR_pred(family, prop, test_size)
    plot(train, test, iteration, fraction, model, prop, family)  ###make plots
    return