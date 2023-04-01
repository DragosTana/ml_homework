import models.kernel_regression as ker_reg
import models.knn as knn
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import misc

def main():
    
    y, x = misc.data_generating_process(1000)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
    mse = []
    bw = []
    n = []
    
    kf = KFold(n_splits = 10)
    b = 0.5
    neigh = 50
    # K-fold cross validation
    
    
    for i, (train_index, validation_index) in enumerate(kf.split(x_train)):
        
        #split data into train and validation
        x_train_kf = [x_train[i] for i in train_index]
        y_train_kf = [y_train[i] for i in train_index]
        x_validation_kf = [x_train[i] for i in validation_index]
        y_validation_kf = [y_train[i] for i in validation_index]
        
        
        prediction = []
        
        # "train" model
        #reg = ker_reg.kernel_regression(x_train_kf, y_train_kf, type = "nadaraya_watson", kernel = "gaussian", bandwidth = b)
        #_, err = pred_ker(reg, x_validation_kf, y_validation_kf)
        
        kn = knn.knn(x_train_kf, y_train_kf, k = neigh)
        _, err = kn.predict(x_validation_kf, y_validation_kf)
        
        mse.append(err)
        n.append(neigh)
        neigh = neigh - 5
        #b = b - 0.05
        #bw.append(b)
        
    plt.plot(n, mse)
    plt.show()
    

def prova_knn():
    
    y, x = misc.data_generating_process(1000)
    
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
    knn_reg = knn.knn(x_train, y_train, 10)
    pred = []
    
    for x in x_test:
        pred.append(knn_reg(x))
        
    plt.plot(x_test, pred, "o", color = "red")
    plt.plot(x_test, y_test, "o", color = "green")
    plt.show()
    
if __name__ == "__main__":
    main()
    