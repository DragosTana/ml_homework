import models.kernel_regression as reg
import models.knn as knn
import misc 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor as KNN
import numpy as np

from matplotlib import pyplot as plt



# TODO: 
# 1) montecarlo simulation to compare the two methods
# 2) check behaviour in homoschedacity and heteroschedacity
# 3) corner behaviour

def demo1():
    
    #generate data with data_generating_process function in misc.py, you can modify the function to generate data with different features
    X, Y = misc.data_generating_process(dimensions = 1000, feature = 2)
    
    #uncomment the following lines to plot the data:
    
    #plt.plot(Y, X, 'o')
    #plt.show()
    
    #split data into train and test using the train_test_split function from sklearn.model_selection. 0.2 means 20% of the data is used for testing
    x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size = 0.2)
    
    # reshape data to be compatible with the sklearn framework
    x_train = np.array(x_train).reshape(-1,1)
    x_test = np.array(x_test).reshape(-1,1)
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)

    
    #create kernel regression and knn regression objects
    ker_reg = reg.KernelRegression(kernel_type = "gaussian", bandwidth = 0.5)
    knn_reg = KNN(n_neighbors = 10)
    
    #use fit function to fit the model to the training data
    ker_reg.fit(x_train, y_train)
    knn_reg.fit(x_train, y_train)
    
    #use predict function to predict the test data
    pred_ker = ker_reg.predict(x_test)
    pred_knn = knn_reg.predict(x_test)
    
    
    plt.plot(x_test, y_test, 'o', label = "test data", color = "red")
    plt.plot(x_test, pred_ker,  'o', label = "kernel regression", color = "blue")
    plt.plot(x_test, pred_knn, 'o', label = "knn regression", color = "green")
    plt.legend()
    plt.show()    
    
    
def demo2():
    
    #generate data with data_generating_process function in misc.py, you can modify the function to generate data with different features
    X, Y = misc.data_generating_process(dimensions = 1000, feature = 2)
    
    #uncomment the following lines to plot the data:
    
    #plt.plot(Y, X, 'o')
    #plt.show()
    
    #split data into train and test using the train_test_split function from sklearn.model_selection. 0.2 means 20% of the data is used for testing
    x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size = 0.2)
    
    # reshape data to be compatible with the sklearn framework
    x_train = np.array(x_train).reshape(-1,1)
    x_test = np.array(x_test).reshape(-1,1)
    #y_train = np.array(y_train).reshape(-1,1)
    #y_test = np.array(y_test).reshape(-1,1)
    
    
    #Here we plot the MSE of the knn regression as a function of the bandwidth, UNCOMMENT THE FOLLOWING LINES TO SEE THE PLOT
    
    #error = []
    #
    #for k in range(1, 100):
    #    knn_reg = KNN(n_neighbors = k)
    #    knn_reg.fit(x_train, y_train)
    #    pred_knn = knn_reg.predict(x_test)
    #    error.append(misc.MSE(y_test, pred_knn))
    #    
    #plt.plot(range(1, 100), error)
    #plt.show()
    
    
    #Here we plot the MSE of the kernel regression as a function of the bandwidth, UNCOMMENT THE FOLLOWING LINES TO SEE THE PLOT
    
    error = []
    
    for k in np.arange(0.02, 1, 0.02):
        ker_reg = reg.KernelRegression(kernel_type = "gaussian", bandwidth = k)
        ker_reg.fit(x_train, y_train)
        pred_knn = ker_reg.predict(x_test)
        error.append(misc.MSE(y_test, pred_knn))
        
    plt.plot(np.arange(0.02, 1, 0.02), error)
    plt.show()
    
    
if __name__ == "__main__":
    demo2()
    
    
    
    
    
    
    
    
    
    
    
    
    #kf = KFold(n_splits = 10)
    #for i, (train_index, validation_index) in enumerate(kf.split(x_train)):
    #    
    #    #split data into train and validation
    #    x_train_kf = [x_train[i] for i in train_index]
    #    y_train_kf = [y_train[i] for i in train_index]
    #    x_validation_kf = [x_train[i] for i in validation_index]
    #    y_validation_kf = [y_train[i] for i in validation_index]