import database
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def bootstrap(prop, iteration, fraction, family, model):
    """
    This function is used to validate model by resampling method, it will generate mse and r2 average.
    """
    data = database.Database()  ###load dataframe
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
