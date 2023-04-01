import numpy as np
    
    
class knn:
    
    def __init__(self, feature, values, k = 5):
        """
        feature: list of features
        values: list of values
        k: number of neighbors
        """
        self.feature = np.array(feature)
        self.values = np.array(values)
        self.k = k
    
    def __call__(self, x):
        euclidean_distance = [np.linalg.norm(x - v) for v in self.feature]
        idx = np.argsort(euclidean_distance)
        return np.mean(self.values[idx[:self.k]])
        