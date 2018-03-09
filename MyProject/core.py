import pandas as pd
import database
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from neupy import algorithms, layers, environment, estimators
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor


def Database():
    """compile data sets into a data library, the output will be a DataFrame"""
    data_1 = pd.read_excel('data/Flash Point and Cetane Number Predictions for Fuel Compounds.xls', skiprows=3) ###load data
    data = data_1[['Name','Family', 'FP Exp.', 'CN Exp.']]  ###select columns
    result_1 = data.drop(index=0)   ###rearrange the index
    result_1.reset_index(drop=True, inplace=True)
    names = ['Name','Family', 'Flash Point', 'Cetane Number']   ###rename columns
    result_1.columns = names
    data_2 = pd.read_excel('data/Flash Point and Cetane Number Predictions for Fuel Compounds.xls', skiprows=4)
    result_2 = data_2.loc[: , '-H': 'aaCa']   ###select specific columns
    smarts = ["[H]", "[CX4H3]", "[CX4H2]", "[CX4H1]", "[CX4H0]", "[CX3H2]", "[CX3H1]", "[CX3H0]", "[CX2H1]","[CX2H0]", "[CX4H2R]", "[CX4H1R]", "[CX4H0R]","[CX3H1R]","[CX3H0R]","[cX3H1](:*):*", "[cX3H0](:*)(:*)*", "[OX2H1]", "[OX2H1][cX3]:[c]", "[OX2H0]", "[OX2H0R]", "[oX2H0](:*):*", "[CX3H0]=[O]", "[CX3H0R]=[O]", "[CX3H1]=[O]", "[CX3H0](=[O])[OX2H1]", "[CX3H0](=[O])[OX2H0]", "[cX3H0](:*)(:*):*"]   ###rename functional groups to SMARTS
    result_2.columns = smarts
    result = pd.concat([result_1, result_2], axis=1)   ###combine two dataframes into one dataframe
    return result

def df_prediction(family, prop, test_size):
    """
    This function is used to create train and test data for prediction.
    """
    data = Database()   ###load, select, clear NaN data
    data_f = data[data.Family == family]
    df = data_f[np.isfinite(data_f[prop])]
    train, test = train_test_split(df, test_size=test_size)  ###split data
    return train, test

def model_selection(family, prop, iteration, fraction, test_size):
    """
    This model is used to select the best model to predict properties according to the minimal mse, the models include
    Ordinary Least Squares, Partial Least Squares, Polynomial Regression, Artificial Neural Network.
    """
    OLS = OLS_pred(family, prop, test_size)[0]  ###build OLS model
    PLS = PLS_pred(family, prop, test_size)[0]  ###build PLS model
    models = [OLS, PLS]
    mse_model = []
    mses_model = []
    mse = []
    mses = []
    for model in models:
        for i in np.arange(1, iteration+1):   ###find the minimal mse during iteration in a specific model
            mse_model = bootstrap(prop, i, fraction, family, model)[0]
            mses_model.append(mse)         
        mse = min(mses_model)
        mses.append(mse)  ###add the minimal mse to the list
    best = mses.index(min(mses))  ###find the minimal mse in all models
    return models[best]    ###return the best model to predict

def plot(train, test, iteration, fraction, model, prop, family):
    """
    This function is used to make parity plot, mse vs. bootstrap samples, r_2 vs. bootstrap samples.
    """
    X_train = train[train.columns[4:]]  ###select functional groups
    X_test = test[test.columns[4:]]
    n = np.arange(1, iteration+1)       ###set bootstrap sample
    mses = []    ###place mse
    r2s = []     ###place r2
    
    fig = plt.figure(figsize=(18, 6))   ###adjust the figsize
    plt.subplot(131)                    ###make parity plot
    plt.scatter(train[prop], model.predict(X_train), label='training data')
    plt.scatter(test[prop], model.predict(X_test), color='r', label='testing data')
    plt.xlabel(prop, fontsize=16)
    plt.ylabel(prop, fontsize=16)
    plt.legend(fontsize=16)
    plt.title('Parity Plot', fontsize=16)
    
    for i in n:
        mse, r2 = bootstrap(prop, i, fraction, family, model)  ###get mse and r2 average for different samples
        mses.append(mse)
        r2s.append(r2)
    
    plt.subplot(132)    ###make mse vs. bootstrap samples
    plt.plot(n, mses)
    plt.xlabel('Number of Bootstrap Samples', fontsize=16)
    plt.ylabel('Mean Squared Error (MSE)', fontsize=16)
    plt.title('Number of Bootstrap Samples vs. MSE', fontsize=16)
    
    plt.subplot(133)    ###make r2 vs. bootstrap samples
    plt.plot(n, r2s)
    plt.xlabel('Number of Bootstrap Samples', fontsize=16)
    plt.ylabel('$R^2$', fontsize=16)
    plt.title('Number of Bootstrap Samples vs. $R^2$', fontsize=16)
    return fig

def bootstrap(prop, iteration, fraction, family, model):
    """
    This function is used to validate model by resampling method, it will generate mse and r2 average.
    """
    data = Database()  ###load dataframe
    data_f = data[data.Family == family]  ###select specific family to resample
    df = data_f[np.isfinite(data_f[prop])]  ###clear the data is NaN
    mse = []   ###place mse values
    r2 = []    ###place r2 values
    for i in range(iteration):
        sample = df.sample(frac=fraction, replace=True)   ###sample from data
        X = sample[sample.columns[4:]]  ###select functional groups
        y = sample[prop]     ###select properties
        y_pred = model.predict(X)   ###build model and predict
        mse.append(mean_squared_error(y, y_pred))   ###compute mse
        r2.append(r2_score(y, y_pred))   ###compute r2
    mse_avg = np.mean(mse)    ###average mse
    r2_avg = np.mean(r2)      ###average r2
    return mse_avg, r2_avg

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

def PLS_pred(family, prop, test_size):
    """
    This function is used to predict properties according to Partial Least Squares(linear model).
    """
    train, test = df_prediction(family, prop, test_size)  ###create data for train and test
    pls = PLSRegression()          ###build model
    train_X = train[train.columns[4:]]   ###select functional groups
    n = len(train_X.columns)    ###the max number of components
    param_grid = [{'n_components' : range(1, n+1)}]   ###set the range of parameter
    PLS = GridSearchCV(PLSRegression(), param_grid)   ###create model
    PLS.fit(train_X, train[prop])   ###modify the model by searching for best n_components
    return PLS, train, test   ###return model, train, test data to plot

def PLS_plot(family, prop, iteration, fraction, test_size):
    """
    This function is used to make plots according to PLS model.
    """
    model, train, test = PLS_pred(family, prop, test_size)
    plot(train, test, iteration, fraction, model, prop, family)  ###make plots
    return

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

def GRNN(family, prop, test_size):
    """This function is used to predict properties by using the General Regression Neural Network model."""
    train, test = df_prediction(family, prop, test_size)  ###create data for train and test
    x_train = train[train.columns[4:]]   ###select functional groups
    y_train = train[prop]

    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(x_train)
    np.set_printoptions(precision=4) # summarize transformed data for x,, and also set up the descimal place of the value
    
    grnn = algorithms.GRNN(std=0.3,verbose=False,)
    grnn.train(x_train, y_train)
    return grnn, train, test

def MLPR(family, prop, test_size):
    """This function is used to predict properties by using the Multiple Layers Perception Regression model."""
    # Input data and define the parameters
    train, test = df_prediction(family, prop, test_size)
    x_train = train[train.columns[4:]]
    y_train = train[prop]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaledX = scaler.fit_transform(x_train)
    np.set_printoptions(precision=4) # summarize transformed data for x,, and also set up the descimal place of the value
        
    mlpr = MLPRegressor(hidden_layer_sizes=(1000,),activation='identity', solver='sgd', learning_rate='adaptive', max_iter=4000, verbose=False)
    mlpr.fit(x_train, y_train)
    return mlpr, train, test
