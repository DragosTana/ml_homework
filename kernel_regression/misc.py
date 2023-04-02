import numpy as np


def data_generating_process(dimensions = 100, feature =  2):
    
    if feature == 1:
        X = np.random.uniform(0, 20, dimensions)
        Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 20) for x in X]
        return(Y,X)
    
    if feature == 2:
        X, Y = np.random.uniform(-20, 20, dimensions), np.random.uniform(-20, 20, dimensions)
        Z = [(X ** 2)*np.cos(X) + (Y **2)*np.cos(Y) + np.random.normal(0, 10) for X, Y in zip(X, Y)]
        return(X, Y, Z)

def input_DGP(function, dimensions):
    """
    function: string
    dimensions: int
    
    input_DGP takes a string and an integer as input and returns a tuple of arrays caluculated from the user defined function.
    """
    feature = int(input("Enter the number of features: "))
    if feature == 1:
        X = np.random.uniform(0, 20, dimensions)
        Y = eval(X, function)
        return(Y,X)
    
    if feature == 2:
        X, Y = np.random.uniform(-20, 20, dimensions), np.random.uniform(-20, 20, dimensions)
        Z = eval(X, Y, function)
        return(X, Y, Z)
    
def MSE(y, y_pred):
    value = float(0)
    for i in range(len(y)):
        tmp = (y[i] - y_pred[i])**2
        value = value + tmp
        
    return value/len(y)

def RMSE(y, y_pred):
    return np.sqrt(MSE(y, y_pred))

def cross_validation():
    pass