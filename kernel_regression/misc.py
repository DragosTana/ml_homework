import numpy as np
import csv


def data_generating_process(dimensions = 100, feature =  2):
    
    if feature == 1:
        X = np.random.uniform(0, 20, dimensions)
        X = np.sort(X)
        Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 20 + 5*x) for x in X]
        Y2 = [(x ** 2)*np.cos(x) for x in X]
        return(Y, Y2, X)
    
    if feature == 2:
        X = np.random.uniform(0, 20, dimensions)
        X = np.sort(X)
        Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 20 ) for x in X]
        Y2 = [(x ** 2)*np.cos(x) for x in X]
        return(Y, Y2, X)
    
    if feature == 3:

        mu, sigma = 0, 2
        mu2, sigma2 = 17, 2
        X1 = np.random.normal(mu, sigma, int(dimensions/2))
        X2 = np.random.normal(mu2, sigma2, int(dimensions/2))
        X = np.concatenate([X1, X2])
        X = np.sort(X)
        Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 40) for x in X]
        Y2 = [(x ** 2)*np.cos(x) for x in X]
        
        return(Y, Y2, X)

    if feature == 4:
        X = np.random.uniform(0, 20, dimensions)
        X = np.sort(X)
        Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 20) for x in X]
        Y2 = [(x ** 2)*np.cos(x) for x in X]
        return(Y, Y2, X)
    
    if feature == 5:
        
        X = np.random.uniform(0, 20, dimensions)
        X = np.sort(X)
        Y = [(3 * x + np.random.normal(0, 3 + 0.5*x)) for x in X]
        Y2 = [(x ** 2)*np.cos(x) for x in X]
        return(Y, Y2, X)
   
def load_csv(file_name):
    data = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            point = []
            for word in line:
                point.append(float(word))
            data.append(point)
    return data

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