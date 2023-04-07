import numpy as np

def data_generating_process(dimension):
    # generate data

    X = np.random.uniform(0, 20, dimension)
    Y = 3*X + np.random.normal(0, 5, dimension)
    return X, Y