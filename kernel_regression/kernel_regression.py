
import numpy as np
import kernel as ker


class kernel_regression:
    def __init__(self, feature, values, type = "nadaraya_watson", kernel = ker.kernel("gaussian"), bandwidth = 1):
        self.kernel = kernel
        self.feature = feature
        self.values = values
        self. bandwidth = bandwidth
        self.type = type
    
    def __call__(self, x):
        
        if self.type == "nadaraya_watson":
            
            #TODO: optimize
            tmp = [x - v for v in self.feature]
            ker_values = [(1/self.bandwidth)*self.kernel(v/self.bandwidth) for v in tmp]

            ker_values = np.array(ker_values)
            values = np.array(self.values)

            num = np.dot(ker_values.T, values)
            denom = np.sum(ker_values)

            return num/denom
        
        if self.type == "priestley_chao":
            pass
    