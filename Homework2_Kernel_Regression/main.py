import models.kernel_regression as reg
import models.knn as knn
import misc 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt

def montecarlo():
    
    montecarslo_simulation = 300
    error_knn = []
    error_reg = []
    
    #MONTECARLO SIMULATION
    for i in tqdm (range (montecarslo_simulation), desc="Loading..."):
        
        #generate data for every simulation
        X, Y = misc.data_generating_process(dimensions = 500, feature = 1)
        
        #split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size = 0.2)
        x_train = np.array(x_train).reshape(-1,1)
        x_test = np.array(x_test).reshape(-1,1)
        
        
        #grid search for best parameters for both KNN and Kernel Regression
        knn = KNN()
        param_grid_knn = { "n_neighbors": np.arange(1, 100) }
        knn_gscv = GridSearchCV(knn, param_grid_knn, cv=5, scoring="neg_mean_squared_error")
        knn_gscv.fit(x_train, y_train)
        
        
        #p1 = mp.Process(target= GridSearchCV(knn, param_grid_knn, cv=5, scoring="neg_mean_squared_error").fit(x_train, y_train))
        
        ker2 = reg.KernelRegression()
        param_grid = { "bandwidth": np.arange(0.02, 0.4, 0.02) }
        ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
        ker_gscv.fit(x_train, y_train)
        
        #p2 = mp.Process(target= GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error").fit(x_train, y_train))
        
        #get best estimator for both KNN and Kernel Regression
        knn_best = knn_gscv.best_estimator_
        ker_best = ker_gscv.best_estimator_
        
        #predict values for both KNN and Kernel Regression
        knn_pred = knn_best.predict(x_test)
        ker_pred = ker_best.predict(x_test)
        
        error_knn.append(mean_squared_error(y_test, knn_pred))
        error_reg.append(mean_squared_error(y_test, ker_pred))
    
    error_knn = np.asarray(error_knn)
    error_reg = np.asarray(error_reg)
    np.savetxt("error_knn_1000.csv", error_knn, delimiter=",")
    np.savetxt("error_reg_1000.csv", error_reg, delimiter=",")
    
def montecarlo_kernel():
    montecarslo_simulation = 100
    error_knn = []
    error_reg = []
    kernel = {"gaussian": 0, "epanechnikov": 0, "uniform": 0, "triangular": 0, "cosine": 0}
    
    #MONTECARLO SIMULATION
    for i in tqdm (range (montecarslo_simulation), desc="Loading..."):
        
        #generate data for every simulation
        X, Y = misc.data_generating_process(dimensions = 200, feature = 1)
        
        #split data into train and test
        x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size = 0.2)
        x_train = np.array(x_train).reshape(-1,1)
        x_test = np.array(x_test).reshape(-1,1)
        
        ker2 = reg.KernelRegression()
        param_grid = { "kernel_type": ["gaussian", "epanechnikov", "uniform", "triangular", "cosine"] }
        ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
        ker_gscv.fit(x_train, y_train)
        
        ker_best = ker_gscv.best_estimator_
        
        ker_pred = ker_best.predict(x_test)
        
        kernel[ker_best.kernel_type] += 1
        
    plt.bar(range(len(kernel)), list(kernel.values()), align='center')
    plt.xticks(range(len(kernel)), list(kernel.keys()))
    plt.show()

def montecarlo_eteroschedacity():
        error_omo = []
    
    #generate data for every simulation
        X_omo, Y_omo = misc.data_generating_process(dimensions = 500, feature = 1)
        
        #split data into train and test
        x_train_omo, x_test_omo, y_train_omo, y_test_omo = train_test_split(Y_omo, X_omo, test_size = 0.2)
        x_train_omo = np.array(x_train_omo).reshape(-1,1)
        x_test_omo = np.array(x_test_omo).reshape(-1,1)
        
        ker2 = reg.KernelRegression()
        param_grid = { "bandwidth": np.arange(0.02, 0.4, 0.02) }
        ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
        ker_gscv.fit(x_train_omo, y_train_omo)
        
        #p2 = mp.Process(target= GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error").fit(x_train, y_train))
        
        #get best estimator for both KNN and Kernel Regression

        ker_best = ker_gscv.best_estimator_
        
        #predict values for both KNN and Kernel Regression

        ker_pred = ker_best.predict(x_test_omo)
        
        error_omo.append(mean_squared_error(y_test_omo, ker_pred))

def nonomogenity():
    
    y, _, x = misc.data_generating_process(dimensions= 500, feature=3)
    x = np.array(x).reshape(-1, 1)
    _, y_func, x_func = misc.data_generating_process(dimensions = 500, feature=2)

    ker2 = reg.KernelRegression()
    param_grid = { "bandwidth": np.arange(0.05, 1, 0.05) }
    ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
    ker_gscv.fit(x, y)
    x_pred = np.array(np.arange(-5, 22, 0.1)).reshape(-1,1)
    y_pred = ker_gscv.predict(x_pred)
    
    plt.plot(x, y, ".")
    plt.plot(x_func, y_func, label = "function",linewidth=1.5)
    plt.plot(x_pred, y_pred, label = "kernel regression", linewidth=1.5)
    plt.legend()
    
    plt.show()
    
    
def heteroschedacity():
    
    #generate data from non uniform distribution:
    
    y, _, x = misc.data_generating_process(dimensions= 1000, feature=5)
    x = np.array(x).reshape(-1, 1)
    _, y_func, x_func = misc.data_generating_process(dimensions = 500, feature=2)

    
    ker2 = reg.KernelRegression()
    param_grid = { "bandwidth": np.arange(0.05, 1, 0.05) }
    ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
    ker_gscv.fit(x, y)
    x_pred = np.array(np.arange(0, 20, 0.1)).reshape(-1,1)
    y_pred = ker_gscv.predict(x_pred)
    
    plt.plot(x, y, ".")
    plt.plot(x_func, y_func, label = "function",linewidth=1.5)
    plt.plot(x_pred, y_pred, label = "kernel regression", linewidth=1.5)
    plt.legend()
    
    plt.show()
    
    
def omoschedacity():
    
    #generate data from non uniform distribution:
    
    y, _, x = misc.data_generating_process(dimensions= 500, feature=4)
    x = np.array(x).reshape(-1, 1)
    _, y_func, x_func = misc.data_generating_process(dimensions = 500, feature=2)

    ker2 = reg.KernelRegression()
    param_grid = { "bandwidth": np.arange(0.05, 1, 0.05) }
    ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
    ker_gscv.fit(x, y)
    x_pred = np.array(np.arange(0, 20, 0.1)).reshape(-1,1)
    y_pred = ker_gscv.predict(x_pred)
    
    plt.plot(x, y, ".")
    plt.plot(x_func, y_func, label = "function",linewidth=1.5)
    plt.plot(x_pred, y_pred, label = "kernel regression", linewidth=1.5)
    plt.legend()
    
    plt.show()

     
def demo1():
    
    #generate data with data_generating_process function in misc.py, you can modify the function to generate data with different features
    _, Y3, X3 = misc.data_generating_process(dimensions = 1000, feature = 2)
    Y, Y2, X = misc.data_generating_process(dimensions=1000, feature=3)
    
    #uncomment the following lines to plot the data:
    
    plt.plot(X, Y, '.', c = "red")
    plt.plot(X3, Y3, c = "blue")
    #plt.show()
    
    #split data into train and test using the train_test_split function from sklearn.model_selection. 0.2 means 20% of the data is used for testing
    x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size = 0.2)
    
    # reshape data to be compatible with the sklearn framework
    x_train = np.array(x_train).reshape(-1,1)
    x_test = np.array(x_test).reshape(-1,1)
    x_test = np.sort(x_test)
    x_train = np.sort(x_train)
    
    #create kernel regression and knn regression objects
    ker_reg = reg.KernelRegression(kernel_type = "gaussian", bandwidth = 0., reg_type= "nadaraya_watson")
    #knn_reg = KNN(n_neighbors = 10)
    
    #use fit function to fit the model to the training data
    ker_reg.fit(x_train, y_train)
    #knn_reg.fit(x_train, y_train)
    
    #use predict function to predict the test data
    pred_ker = ker_reg.predict(np.arange(-10, 30, 1).reshape(-1,1))
    #pred_knn = knn_reg.predict(x_test)
    
    
    #plt.plot(x_test, y_test, 'o', label = "test data", color = "red")
    plt.plot(np.arange(-10, 30, 1), pred_ker, label = "kernel regression", color = "green")
    #plt.plot(x_test, pred_knn, 'o', label = "knn regression", color = "green")
    plt.legend()
    plt.show()    
    
    
def demo2():
    
    #generate data with data_generating_process function in misc.py, you can modify the function to generate data with different features
    X, Y = misc.data_generating_process(dimensions = 200, feature = 1)
    
    #uncomment the following lines to plot the data:
    
    #plt.plot(Y, X, 'o')
    #plt.show()
    
    #split data into train and test using the train_test_split function from sklearn.model_selection. 0.2 means 20% of the data is used for testing
    x_train, x_test, y_train, y_test = train_test_split(Y, X, test_size = 0.2)
    
    # reshape data to be compatible with the sklearn framework
    x_train = np.array(x_train).reshape(-1,1)
    x_test = np.array(x_test).reshape(-1,1)

    
    #Here we plot the MSE of the knn regression as a function of the bandwidth, UNCOMMENT THE FOLLOWING LINES TO SEE THE PLOT

    #error = []
    #
    #for k in range(1, 100):
    #    knn_reg = KNN(n_neighbors = k)
    #    knn_reg.fit(x_train, y_train)
    #    pred_knn = knn_reg.predict(x_test)
    #    error.append(misc.MSE(y_test, pred_knn))
    #    
    #knn2 = KNN()
    ##create a dictionary of all values we want to test for n_neighbors
    #param_grid = { "n_neighbors": np.arange(1, 100) }
    ##use gridsearch to test all values for n_neighbors
    #knn_gscv = GridSearchCV(knn2, param_grid, cv=5, score="neg_mean_squared_error")
    ##fit model to data
    #knn_gscv.fit(x_train, y_train)
    #print(knn_gscv.best_params_.values())
    #
    #plt.plot(range(1, 100), error)
    #plt.show()
    
    #Here we plot the MSE of the kernel regression as a function of the bandwidth, UNCOMMENT THE FOLLOWING LINES TO SEE THE PLOT
    
    error = []
    
    for k in np.arange(0.02, 0.4, 0.02):
        ker_reg = reg.KernelRegression(kernel_type = "gaussian", bandwidth = k)
        ker_reg.fit(x_train, y_train)
        pred_knn = ker_reg.predict(x_test)
        error.append(misc.MSE(y_test, pred_knn))
    
    ker2 = reg.KernelRegression()
    param_grid = { "bandwidth": np.arange(0.02, 0.4, 0.02) }
    ker_gscv = GridSearchCV(ker2, param_grid, cv=5, scoring="neg_mean_squared_error")
    ker_gscv.fit(x_train, y_train)
   
    ker_best = ker_gscv.best_estimator_
    
    ker_pred = ker_best.predict(x_test)
    
    mse = mean_squared_error(y_test, ker_pred)
    print(mse)
    print(ker_gscv.best_params_)
    
    plt.plot(np.arange(0.02, 0.4, 0.02), error)
    plt.show()

    
if __name__ == "__main__":
    heteroschedacity()