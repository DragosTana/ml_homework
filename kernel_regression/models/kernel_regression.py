
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

import models.kernel as ker
        
    
class KernelRegression(BaseEstimator, RegressorMixin):
    """
    kernel_type: "gaussian", "uniform", "triangular", "epanechnikov", "cosine"
    bandwidth: float
    
    Kernel regression model compatible with sklearn. 
    """

    def __init__(self, kernel_type  = "gaussian", bandwidth = 1.0, reg_type = "nadaraya_watson"):
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type
        self.kernel = ker.Kernel(kernel_type)
        self.reg_type = reg_type
        
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        self.X_ = X
        self.y_ = y
        return self
    
    def predict(self, X):
        
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        
        if self.reg_type == "nadaraya_watson":
            pred = []
            for x in X:
                tmp = [x - v for v in self.X_]
                ker_values = [(1/self.bandwidth)*self.kernel(v/self.bandwidth) for v in tmp]

                ker_values = np.array(ker_values)
                values = np.array(self.y_)

                num = np.dot(ker_values.T, values)
                denom = np.sum(ker_values)

                pred.append(num/denom)
            return pred

           
        
        if self.reg_type == "Priestley_Chao":
            pass
        