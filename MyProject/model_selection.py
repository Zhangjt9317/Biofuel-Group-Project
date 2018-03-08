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