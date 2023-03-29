import kernel as ker
import kernel_regression as ker_reg
from matplotlib import pyplot as plt
import numpy as np
import random

def data_generating_process(dimensions):
    X = np.random.uniform(0, 20, dimensions)
    Y = [x ** 2 + np.random.normal(0, 5) for x in X]
    return(Y,X)

def main():
  
    y, x = data_generating_process(200)
    
    dati = []
    for i in range(len(y)):
        coppia = (x[i], y[i])
        dati.append(coppia)
        
    dati = random.shuffle(dati)
    
    train = []
    test = []
    l = 200
    
    for i in range(int(l*0.7)):
        train = dati[i]
        
    for i in range(int(l*0.7), l):
        test = dati[i]
 
    x_train = []
    y_train = []
    for e in train:
        x_train.append(e[0])
        y_train.append(e[1])
        
    x_test = []
    for e in test:
        x_test.append(e[0])
        
        
    gauss = ker.kernel("gaussian")
    reg = ker_reg.kernel_regression(x_train, y_train, type="nadaraya_watson", kernel=gauss, bandwidth=5)
    
   
    values = []
    
    for i in range(len(test)):
        value = reg(test[i][0])
        values.append(value)
        

    
    
    
    plt.plot(x_test, values, 'o', c = 'red')
    plt.plot(x_train, y_train, 'o')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()