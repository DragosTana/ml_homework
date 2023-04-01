import models.kernel as ker
import models.kernel_regression as ker_reg
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import misc


def regression2D():
    
    y, x = misc.data_generating_process(5000)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
    mse = []
    bw = []
    
    kf = KFold(n_splits = 10)
    b = 0.5
    # K-fold cross validation
    
    
    for i, (train_index, validation_index) in enumerate(kf.split(x_train)):
        
        #split data into train and validation
        x_train_kf = [x_train[i] for i in train_index]
        y_train_kf = [y_train[i] for i in train_index]
        x_validation_kf = [x_train[i] for i in validation_index]
        y_validation_kf = [y_train[i] for i in validation_index]
        
        
        prediction = []
        
        # "train" model
        reg = ker_reg.kernel_regression(x_train_kf, y_train_kf, type = "nadaraya_watson", kernel = "gaussian", bandwidth = b)
        for i in range(len(y_validation_kf)):
            p = reg(x_validation_kf[i])
            prediction.append(p)
        mse.append(misc.MSE(y_validation_kf, prediction))
        b = b - 0.05
        bw.append(b)
        
    plt.plot(bw, mse)
    plt.show()
    

def regression3D():
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    
    x, y, z = misc.data_generating_process(2000, feature = 3)
    
    ax.scatter(x, y, z, 'o', c = z)
    plt.show()

regression2D()