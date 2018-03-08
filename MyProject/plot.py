import matplotlib.pyplot as plt

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
