def df_prediction(family, prop, test_size):
    """
    This function is used to create train and test data for prediction.
    """
    data = Database()   ###load, select, clear NaN data
    data_f = data[data.Family == family]
    df = data_f[np.isfinite(data_f[prop])]
    train, test = train_test_split(df, test_size=test_size)  ###split data
    return train, test
