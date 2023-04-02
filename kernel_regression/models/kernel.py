import numpy as np
            

class Kernel:
    def __init__(self, kernel_type = "gaussian"):
        self.kernel_type = kernel_type
              
    def __call__(self, value):
        match self.kernel_type:
            case "gaussian":
                return np.exp(-value**2/(2))/(np.sqrt(2*np.pi))
            case "uniform":
                return np.where(np.abs(value) <= 1, 1/2, 0)
            case "triangular":
                return np.where(np.abs(value) <= 1, 1 - np.abs(value), 0)
            case "epanechnikov":
                return np.where(np.abs(value) <= 1, 3/(4)*(1 - (value)**2), 0)
            case "cosine":
                return np.where(np.abs(value) <= 1, np.pi/(4)*np.cos(np.pi*value), 0)
    