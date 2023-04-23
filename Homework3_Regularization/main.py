import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix

def main():
    
    #parameters 
    features = 6
    size = 1000
    beta = [0, 2, -1, 5, 5, 0]
    
    #generate covariance matrix 
    mean = np.random.uniform(-5, 5, size = features).astype(int)
    cov = 4*datasets.make_spd_matrix(features)
        
    #data generating process
    X = np.random.multivariate_normal(mean, cov, size=size)
    Y = np.dot(X, beta) + np.random.normal(0, 1, size = size)
    
    plot = False
    if plot == True:
        #transforming data to pandas dataframe and plotting correlation matrix and scatter matrix
        df = pd.DataFrame(X, columns = ["X{i}".format(i=i) for i in range(1, features+1)])
        f = plt.figure()
        plt.matshow(df.corr(), fignum=f.number, cmap='coolwarm')
        plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
        plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)

        scatter_matrix(df, alpha = 0.2,  figsize = (6, 6), diagonal = 'kde')

        plt.show()
    
    linear_reg = LinearRegression()
    ridge_reg = RidgeCV()
    lasso_reg = LassoCV()
    elastic_reg = ElasticNetCV()
    
    linear_reg_coef = []
    ridge_reg_coef = []
    lasso_reg_coef = []
    elastic_reg_coef = []
    
    #MONTECARLO SIMULATION
    montecarlo_simulation = 1000
    
    print(" ")
    print("Running Monte Carlo Simulation...")
    print(" ")
    for i in tqdm (range (montecarlo_simulation), desc="Loading..."):
        X = np.random.multivariate_normal(mean, cov, size=size)
        Y = np.dot(X, beta) + np.random.normal(0, 1, size = size)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
        
        #fitting the models
        linear_reg.fit(x_train, y_train)
        ridge_reg.fit(x_train, y_train)
        lasso_reg.fit(x_train, y_train)
        elastic_reg.fit(x_train, y_train)
        
        linear_reg_coef.append(linear_reg.coef_)
        ridge_reg_coef.append(ridge_reg.coef_)
        lasso_reg_coef.append(lasso_reg.coef_)
        elastic_reg_coef.append(elastic_reg.coef_)  
    
    linear_reg_coef = np.array(linear_reg_coef)
    ridge_reg_coef = np.array(ridge_reg_coef)
    lasso_reg_coef = np.array(lasso_reg_coef)
    elastic_reg_coef = np.array(elastic_reg_coef)
    
    fig, ax = plt.subplots(2, 2)
    ax[0,0].hist(linear_reg_coef[:, 0])
    ax[0,0].set_title("Linear Regression")
    ax[0,1].hist(ridge_reg_coef[:, 0])
    ax[0,1].set_title("Ridge Regression")
    ax[1,0].hist(lasso_reg_coef[:, 0])
    ax[1,0].set_title("Lasso Regression")
    ax[1,1].hist(elastic_reg_coef[:, 0])
    ax[1,1].set_title("Elastic Net Regression")
    plt.show()
    

if __name__== "__main__":
    main()