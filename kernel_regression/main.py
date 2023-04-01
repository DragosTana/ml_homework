import models.kernel_regression as ker_reg
import models.knn as knn
from sklearn.model_selection import KFold

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import misc

# TODO 
# 1) montecarlo simulation to compare the two methods
# 2) check behaviour in homoschedacity and heteroschedacity
# 3) corner behaviour

def main():
    
    montecalro_simulation = 100
    
    for i in range(montecalro_simulation):
        
        y, x = misc.data_generating_process(5000)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 44)
        kf = KFold(n_splits = 10)

    # K FOLD CROSS VALIDATION
    
    for i, (train_index, validation_index) in enumerate(kf.split(x_train)):
        
        #split data into train and validation
        x_train_kf = [x_train[i] for i in train_index]
        y_train_kf = [y_train[i] for i in train_index]
        x_validation_kf = [x_train[i] for i in validation_index]
        y_validation_kf = [y_train[i] for i in validation_index]
        
        
        
     
    

def prova_knn():
    
    y, x = misc.data_generating_process(5000)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
    kf = KFold(n_splits = 20)
    neighbors = 200
    mse = []
    n = []

    for i, (train_index, validation_index) in enumerate(kf.split(x_train)):
        
        #split data into train and validation
        x_train_kf = [x_train[i] for i in train_index]
        y_train_kf = [y_train[i] for i in train_index]
        x_validation_kf = [x_train[i] for i in validation_index]
        y_validation_kf = [y_train[i] for i in validation_index]
        
        knn_model = knn.knn(x_train_kf, y_train_kf, neighbors)
        _, error = knn_model.predict(x_validation_kf, y_validation_kf)
        mse.append(error)
        n.append(neighbors)
        neighbors = neighbors - 10
    
    plt.plot(n, mse, color = "blue")
    plt.show()
    
if __name__ == "__main__":
    prova_knn()
    