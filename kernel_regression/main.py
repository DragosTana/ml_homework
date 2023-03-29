import kernel as ker
import kernel_regression as ker_reg

from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import misc


def main():
  
    y, x = misc.data_generating_process(500)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)
    
        
    gauss = ker.kernel("gaussian")
    reg = ker_reg.kernel_regression(x_train, y_train, type="nadaraya_watson", kernel=gauss, bandwidth=0.25)
    
   
    values = []
   
    
    for i in range(len(y_test)):
        value = reg(x_test[i])
        values.append(value)
        
  
        
    mse = misc.MSE(y_test, values)
    print(mse)    
    plt.plot(x_test, y_test, 'o', c = 'green')
    plt.plot(x_test, values, '.', c = 'red')
    #plt.plot(x_train, y_train, 'o')
    plt.show()
    
    
    
if __name__ == "__main__":
    main()