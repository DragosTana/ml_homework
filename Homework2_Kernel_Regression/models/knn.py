import numpy as np
import misc as misc
    
    
class knn:
    
    def __init__(self, feature, values, k = 5):
        """
        feature: list of features
        values: list of values
        k: number of neighbors
        
        Naive implementation of k-nearest neighbors for regression.
        """
        self.feature = np.array(feature)
        self.values = np.array(values)
        self.k = k
    
    def __call__(self, x):
        euclidean_distance = [np.linalg.norm(x - v) for v in self.feature]
        idx = np.argsort(euclidean_distance)
        return np.mean(self.values[idx[:self.k]])
    
    def predict(self, x_test, y_test):
        pred = []
        for x in x_test:
            pred.append(self.__call__(x))
        mse = misc.MSE(y_test, pred)
        return pred, mse
        