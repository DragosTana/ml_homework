import numpy as np
            

class kernel:
    def __init__(self, kernel_type = "gaussian", param = None):
        self.kernel_type = kernel_type
        self.param = param
        
    def __call__(self, value):
        
        match self.kernel_type:
            case "gaussian":
                return gaussian_kernel(value, **self.param)
            case "uniform":
                return uniform_kernel(value, **self.param)
            case "triangular":
                return triangular_kernel(value, **self.param)
            case "epanechnikov":
                return epanechnikov_kernel(value, **self.param)
            case "cosine":
                return cosine_kernel(value, **self.param)
             
   
    
def gaussian_kernel(value, sigma = 1):
    return np.exp(-value**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

def uniform_kernel(value, a = 1):
    return np.where(np.abs(value) <= a, 1/(2*a), 0)

def triangular_kernel(value, a = 1):
    return np.where(np.abs(value) <= a, 1 - np.abs(value)/a, 0)

def epanechnikov_kernel(value, a = 1):
    return np.where(np.abs(value) <= a, 3/(4*a)*(1 - (value/a)**2), 0)

def cosine_kernel(value, a = 1):
    return np.where(np.abs(value) <= a, np.pi/(4*a)*np.cos(np.pi*value/a), 0)
    