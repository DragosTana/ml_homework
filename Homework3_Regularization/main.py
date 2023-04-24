import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, RidgeCV, LinearRegression, ElasticNetCV
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from dataVisualization import load_csv
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from os.path import exists

def generate_data():
    #parameters 
    features = 6
    size = 1000
    
    #generate covariance matrix 
    mean = np.random.uniform(-5, 5, size = features).astype(int)
    cov = 4*datasets.make_spd_matrix(features)
    
    for e in cov:
        for i in range(len(e)):
            if abs(e[i]) < 4:
                e[i] = 0
                
    #data generating process
    X = np.random.multivariate_normal(mean, cov, size=size)
    
    plot = True
    if plot:
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
    
    np.savetxt("X.csv", X, delimiter=",")
    np.savetxt("mean.csv", mean, delimiter=",")
    np.savetxt("cov.csv", cov, delimiter=",")

    
def main():
    beta = [0, 2, -1, 5, 5, 0]
    
    mean = np.ravel(np.array(load_csv("/home/dragos/Projects/ML_Homework/mean.csv")))
    cov = np.array(load_csv("/home/dragos/Projects/ML_Homework/cov.csv"))
    
    linear_reg = LinearRegression()
    ridge_reg = RidgeCV()
    lasso_reg = LassoCV()
    elastic_reg = ElasticNetCV()
    
    #MONTECARLO SIMULATION
    montecarlo_simulation = 1000
    #sample_num = [100, 200, 500, 1000, 3000, 5000, 10000]
    sample_num = [100]
    
    for n in sample_num:
        
        linear_reg_coef = []
        ridge_reg_coef = []
        lasso_reg_coef = []
        elastic_reg_coef = []
        
        linear_reg_scores = []
        ridge_reg_scores = []
        lasso_reg_scores = []
        elastic_reg_scores = []
        
        for i in tqdm (range (montecarlo_simulation), desc="Running Monte Carlo Simulation..."):

            X = np.random.multivariate_normal(mean, cov, size=n)
            Y = np.dot(X, beta) + np.random.normal(0, 1, size = n)
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
            
            linear_reg_scores.append(linear_reg.score(x_test, y_test))
            ridge_reg_scores.append(ridge_reg.score(x_test, y_test))
            lasso_reg_scores.append(lasso_reg.score(x_test, y_test))
            elastic_reg_scores.append(elastic_reg.score(x_test, y_test))

        linear_reg_coef = np.array(linear_reg_coef)
        ridge_reg_coef = np.array(ridge_reg_coef)
        lasso_reg_coef = np.array(lasso_reg_coef)
        elastic_reg_coef = np.array(elastic_reg_coef)
        
        linear_reg_scores = np.array(linear_reg_scores)
        ridge_reg_scores = np.array(ridge_reg_scores)
        lasso_reg_scores = np.array(lasso_reg_scores)
        elastic_reg_scores = np.array(elastic_reg_scores)
        
        np.savetxt("/home/dragos/Projects/ML_Homework/data/linear_reg_coef_{n}.csv".format(n=n), linear_reg_coef, delimiter=",")
        np.savetxt("/home/dragos/Projects/ML_Homework/data/ridge_reg_coef_{n}.csv".format(n=n), ridge_reg_coef, delimiter=",")
        np.savetxt("/home/dragos/Projects/ML_Homework/data/lasso_reg_coef_{n}.csv".format(n=n), lasso_reg_coef, delimiter=",")
        np.savetxt("/home/dragos/Projects/ML_Homework/data/elastic_reg_coef_{n}.csv".format(n=n), elastic_reg_coef, delimiter=",")

        np.savetxt("/home/dragos/Projects/ML_Homework/data/linear_reg_scores_{n}.csv".format(n=n), linear_reg_scores, delimiter=",")
        np.savetxt("/home/dragos/Projects/ML_Homework/data/ridge_reg_scores_{n}.csv".format(n=n), ridge_reg_scores, delimiter=",")
        np.savetxt("/home/dragos/Projects/ML_Homework/data/lasso_reg_scores_{n}.csv".format(n=n), lasso_reg_scores, delimiter=",")
        np.savetxt("/home/dragos/Projects/ML_Homework/data/elastic_reg_scores_{n}.csv".format(n=n), elastic_reg_scores, delimiter=",")
        
        print(linear_reg_scores)
        print(ridge_reg_scores)
if __name__== "__main__":
    main()
   