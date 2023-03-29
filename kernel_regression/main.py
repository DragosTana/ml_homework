import kernel as ker
import kernel_regression as ker_reg
from matplotlib import pyplot as plt
import numpy as np
import random

def data_generating_process(dimensions):
    X = np.random.uniform(0, 20, dimensions)
    Y = [(x ** 2)*np.cos(x) + np.random.normal(0, 20) for x in X]
    return(Y,X)

def MSE(y, y_pred):
    value = float(0)
    for i in range(len(y)):
        tmp = (y[i] - y_pred[i])**2
        value = value + tmp
        
    return value/len(y)

def main():
  
    y, x = data_generating_process(500)
    
    dati = []
    for i in range(len(y)):
        coppia = (x[i], y[i])
        dati.append(coppia)
    
    
    random.shuffle(dati)

    train = []
    test = []
    l = len(dati)
    
    for i in range(int(l*0.7)):
        train.append(dati[i])
        
    for i in range(int(l*0.7), l):
        test.append(dati[i])
 
    x_train = []
    y_train = []
    for e in train:
        x_train.append(e[0])
        y_train.append(e[1])
        
    x_test = []
    y_test = []
    for e in test:
        x_test.append(e[0])
        y_test.append(e[1])
        
        
        
    gauss = ker.kernel("gaussian")
    reg = ker_reg.kernel_regression(x_train, y_train, type="nadaraya_watson", kernel=gauss, bandwidth=0.25)
    
   
    values = []
    
    for i in range(len(test)):
        value = reg(test[i][0])
        values.append(value)
        
  
        
    mse = MSE(y_test, values)
    print(mse)    
    plt.plot(x_test, y_test, 'o', c = 'green')
    plt.plot(x_test, values, 'o', c = 'red')
    #plt.plot(x_train, y_train, 'o')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()