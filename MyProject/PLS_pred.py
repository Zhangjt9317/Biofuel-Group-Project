from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

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