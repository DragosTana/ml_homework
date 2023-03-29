import numpy as np

def data_generating_process(dimensions):
    X = np.random.uniform(0, 20, dimensions)
    Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 20) for x in X]
    return(Y,X)

def MSE(y, y_pred):
    value = float(0)
    for i in range(len(y)):
        tmp = (y[i] - y_pred[i])**2
        value = value + tmp
        
    return value/len(y)