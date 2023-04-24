import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import csv
from pandas.plotting import scatter_matrix

def load_csv(file_name):
    data = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            point = []
            for word in line:
                point.append(float(word))
            data.append(point)
    return np.array(data)

def calculate_mean(X):
    coef = load_csv("/home/dragos/Projects/ML_Homework/data/"+X)
    coef_mean = []
    for i in range(len(coef[0, :])): 
        coef_mean.append(np.mean(coef[:, i]))
        
    return coef_mean

def calculate_confidence_interval(X):
    data = load_csv("/home/dragos/Projects/ML_Homework/data/"+X)
    ci = []
    for i in range(len(data[0, :])):
        ci.append(stats.norm.interval(0.95, loc=np.mean(data[:, i]), scale=np.std(data[:, i])/np.sqrt(len(data[:, i]))))
        
    return np.array(ci)

def plot_corr_matrix(X):
    
    features = len(X[0])
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

def mean_rsquared(X):
    err = load_csv("/home/dragos/Projects/ML_Homework/data/" + X)
    return np.mean(err)
    
def prova():
    err_e = []
    err_las = []
    err_lin = []
    err_r = []
    sample_num = [100, 200, 500, 1000, 3000, 5000, 10000]
    
    for n in sample_num:
        err_e.append(mean_rsquared("elastic_reg_scores_{n}.csv".format(n=n)))
        err_las.append(mean_rsquared("lasso_reg_scores_{n}.csv".format(n=n)))
        err_lin.append(mean_rsquared("linear_reg_scores_{n}.csv".format(n=n)))
        err_r.append(mean_rsquared("ridge_reg_scores_{n}.csv".format(n=n)))
        
    plt.plot(sample_num, err_e, label = "ElasticNet")
    plt.plot(sample_num, err_las, label = "Lasso")
    plt.plot(sample_num, err_lin, label = "Linear")
    plt.plot(sample_num, err_r, label = "Ridge")
    plt.legend()
    plt.xscale('log')
    plt.show()
        
        


def main():
    
    X = load_csv("/home/dragos/Projects/ML_Homework/X.csv")

    #plot_corr_matrix(X)
    
    sample_num = [100, 200, 500, 1000, 3000, 5000, 10000]
    beta = [0, 2, -1, 5, 5, 0]
    
    e_mean = []
    las_mean = []
    lin_mean = []
    r_mean = []
    
    e_mean_ci = []
    las_mean_ci = []
    lin_mean_ci = []
    r_mean_ci = []
    
    #for each sample size, calculate the mean and confidence interval for each coefficient
    for n in sample_num:
        
        e_mean.append(calculate_mean("elastic_reg_coef_{n}.csv".format(n=n)))
        las_mean.append(calculate_mean("lasso_reg_coef_{n}.csv".format(n=n)))
        lin_mean.append(calculate_mean("linear_reg_coef_{n}.csv".format(n=n)))
        r_mean.append(calculate_mean("ridge_reg_coef_{n}.csv".format(n=n)))

        e_mean_ci.append(calculate_confidence_interval("elastic_reg_coef_{n}.csv".format(n=n)))
        las_mean_ci.append(calculate_confidence_interval("lasso_reg_coef_{n}.csv".format(n=n)))
        lin_mean_ci.append(calculate_confidence_interval("linear_reg_coef_{n}.csv".format(n=n)))        
        r_mean_ci.append(calculate_confidence_interval("ridge_reg_coef_{n}.csv".format(n=n)))
        
    e_mean = np.array(e_mean)
    las_mean = np.array(las_mean)
    lin_mean  = np.array(lin_mean)
    r_mean = np.array(r_mean)
    
    e_mean_ci = np.array(e_mean_ci)
    las_mean_ci = np.array(las_mean_ci)
    lin_mean_ci  = np.array(lin_mean_ci)
    r_mean_ci = np.array(r_mean_ci)
 
    #plotting the mean and confidence interval for each coefficient
    i = 2
    plt.plot(sample_num, e_mean[:, i], label = "ElasticNet")
    plt.fill_between(sample_num, e_mean_ci[:, i, :][:, 0], e_mean_ci[:, i, :][:, 1], alpha=0.2)
    plt.plot(sample_num, las_mean[:, i], label = "Lasso")
    plt.fill_between(sample_num, las_mean_ci[:, i, :][:, 0], las_mean_ci[:, i, :][:, 1], alpha=0.2)
    plt.plot(sample_num, lin_mean[:, i], label = "Linear")
    plt.fill_between(sample_num, lin_mean_ci[:, i, :][:, 0], lin_mean_ci[:, i, :][:, 1], alpha=0.2)
    plt.plot(sample_num, r_mean[:, i], label = "Ridge")
    plt.fill_between(sample_num, r_mean_ci[:, i, :][:, 0], r_mean_ci[:, i, :][:, 1], alpha=0.2)
    plt.axhline(y=beta[i], color='black', linestyle='--')
    plt.xscale('log')
    plt.title("beta{i}".format(i=i+1))
    plt.legend()
    plt.show()
       
if __name__ == "__main__":
    main()