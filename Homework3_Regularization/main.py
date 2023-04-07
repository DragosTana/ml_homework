import matplotlib.pyplot as plt
from sklearn import linear_model

import misc

def main():
    x, y = misc.data_generating_process(100)
    
    reg_ols = linear_model.LinearRegression()
    reg_ols.fit(x.reshape(-1, 1), y)
    
    ridge_reg = linear_model.Ridge(alpha=0.5)
    ridge_reg.fit(x.reshape(-1, 1), y)
    
    coeff_ridge = ridge_reg.coef_
    coeff = reg_ols.coef_
    print(coeff)
    print(coeff_ridge)
    
    
    plt.plot(x, y, 'o')
    plt.show()
    
if __name__== "__main__":
    main()