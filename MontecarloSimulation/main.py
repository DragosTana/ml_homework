import numpy
from matplotlib import pyplot as plt
from sklearn import linear_model as ln
from scipy.stats import norm
import math
import numpy as np
import matplotlib as mpl

mpl.use('TkAgg')


### MY MONTECARLO SIMULATION

def data_generating_process(mu, sigma, obs, seed = None):
    np.random.seed(seed)
    data = np.random.normal(mu, sigma, obs)

    return data

def d_calculation(data, mu, sigma):
    data = np.sort(data)
    d = 0
    for i in np.arange(data[0], data[-1], 0.1):
        fn1 = norm.cdf(i, mu, sigma)
        fn2 = np.searchsorted(data, i, side='right') / data.size

        dtemp = math.fabs(fn2 - fn1)
        if dtemp > d:
            d = dtemp

    return d

def main():

    # SIMULATION I: data is sampled from the distributions we are testing.
    mu = 10
    sigma = 3
    n = 100
    experiments = 500

    d = []

    for i in range(experiments):
        data = data_generating_process(mu, sigma, n)
        d.append(d_calculation(data, mu, sigma))

    threshold = 1.36 / (pow(n, 0.5))

    d = np.sort(d)
    p = np.searchsorted(d, threshold, side='right') / d.size
    print(p)

    plt.title("Experiments: 500, sample size: 100")
    plt.hist(d, 25)
    plt.axvline(x = threshold, color = "red")
    plt.show()

    # SIMULATION II: we are increasing the sample size
    mu = 10
    sigma = 3
    n = 1000
    experiments = 500

    d = []

    for i in range(experiments):
        data = data_generating_process(mu, sigma, n)
        d.append(d_calculation(data, mu, sigma))

    threshold = (1.36)/(pow(n, 0.5))

    d = np.sort(d)
    p = np.searchsorted(d, threshold, side='right') /d.size
    print(p)

    plt.title("Experiments: 500, sample size: 1000")
    plt.hist(d, 25)
    plt.axvline(x = threshold, color = "red")
    plt.show()

    # SIMULATION III: data is sampled from a distribution slightly different from the distribution for which we are testing

    mu = 10
    sigma = 3
    n = 100
    experiments = 500

    d = []

    for i in range(experiments):
        data = data_generating_process(mu, sigma, n)
        d.append(d_calculation(data, 9.9, 2.9))

    threshold = (1.36)/(pow(n, 0.5))

    d = np.sort(d)
    p = np.searchsorted(d, threshold, side='right') /d.size
    print(p)

    plt.title("Experiments: 500, sample size: 100")
    plt.hist(d, 25)
    plt.axvline(x = threshold, color = "red")
    plt.show()
                                                                                 # SIMULATION IV: data is sampled from a distribution slightly different from the distribution for which we are testing

    mu = 10
    sigma = 3
    n = 1000
    experiments = 500

    d = []

    for i in range(experiments):
        data = data_generating_process(mu, sigma, n)
        d.append(d_calculation(data, 9.9, 2.9))

    threshold = (1.36) / (pow(n, 0.5))

    d = np.sort(d)
    p = np.searchsorted(d, threshold, side='right') / d.size
    print(p)

    plt.title("Experiments: 500, sample size: 1000")
    plt.hist(d, 25)
    plt.axvline(x=threshold, color="red")
    plt.show()

    # SIMULATION V: data is sampled from a distribution slightly different from the distribution for which we are testing

    mu = 10
    sigma = 3
    n = 100
    experiments = 500

    d = []

    for i in range(experiments):
        data = data_generating_process(mu, sigma, n)
        d.append(d_calculation(data, 9, 5))

    threshold = (1.36) / (pow(n, 0.5))

    d = np.sort(d)
    p = np.searchsorted(d, threshold, side='right') / d.size
    print(p)

    plt.title("Experiments: 500, sample size: 1000")
    plt.hist(d, 25)
    plt.axvline(x=threshold, color="red")
    plt.show()


### PROPOSED EXERCISE ON MONTECARLO SIMULATION

def dgp(beta, mean, covariance_matrix, error_variance, obs, plot=False):
    x1, x2 = np.random.multivariate_normal(mean, covariance_matrix, obs).T
    y = beta[0] + beta[1] * x1 + beta[2] * x2 + np.random.normal(0, error_variance, obs)
    if plot == True:
        f1 = plt.figure(1)
        plt.plot(x1, x2, '.', alpha=0.5)
    return y, x1, x2


def exercise():
    # COEFFICIENTS
    beta = np.random.randint(-100, 100, size=3)
    print("beta_0: ", beta[0])
    print("beta_1: ", beta[1])
    print("beta_2: ", beta[2])

    # DATA GENERATING PROCESS PARAMETERS
    cov1 = np.array([[1, 0],
                     [0, 1]])

    cov2 = np.array([[1, 0.5],
                     [0.5, 1]])

    mean = np.array([0, 0])

    sample_size = 10000

    # MONTECARLO SIMULATION PARAMETER
    experiments = 1000

    fig, (ax1, ax2) = plt.subplots(1, 2)

    # MONTECARLO SIMULATION: scenario 1 using x1
    beta1_11 = []
    for i in range(experiments):
        y, x1, x2 = dgp(beta, mean, cov1, 1, sample_size)
        reg1 = ln.LinearRegression().fit(x1.reshape(-1, 1), y)
        # print(reg1.coef_, reg1.intercept_)
        beta1_11.append(reg1.coef_[0])

    x = np.sort(beta1_11)
    y = np.arange(len(beta1_11)) / float(len(beta1_11))
    ax1.plot(x, y, color="red")

    # MONTECARLO SIMULATION: scenario 1 using x1 and x2
    beta1_12 = []
    for i in range(experiments):
        y, x1, x2 = dgp(beta, mean, cov1, 1, sample_size)
        reg1 = ln.LinearRegression().fit(np.array([x1, x2]).T, y)
        # print(reg1.coef_, reg1.intercept_)
        beta1_12.append(reg1.coef_[0])

    x = np.sort(beta1_12)
    y = np.arange(len(beta1_12)) / float(len(beta1_12))
    ax1.plot(x, y, color="blue")
    ax1.set_title("Scenario 1")

    # MONTECARLO SIMULATION: scenario 2 using x1
    beta2_21 = []
    for i in range(experiments):
        y, x1, x2 = dgp(beta, mean, cov2, 1, sample_size)
        reg1 = ln.LinearRegression().fit(x1.reshape(-1, 1), y)
        # print(reg1.coef_, reg1.intercept_)
        beta2_21.append(reg1.coef_[0])

    x = np.sort(beta2_21)
    y = np.arange(len(beta2_21)) / float(len(beta2_21))
    ax2.plot(x, y, color="red")

    # MONTECARLO SIMULATION: scenario 1 using x1 and x2
    beta2_22 = []
    for i in range(experiments):
        y, x1, x2 = dgp(beta, mean, cov2, 1, sample_size)
        reg1 = ln.LinearRegression().fit(np.array([x1, x2]).T, y)
        # print(reg1.coef_, reg1.intercept_)
        beta2_22.append(reg1.coef_[0])

    x = np.sort(beta2_22)
    y = np.arange(len(beta2_22)) / float(len(beta2_22))
    ax2.plot(x, y, color="blue")
    ax2.set_title("Scenario 2")

    plt.show()


if __name__ == "__main__":
    main()
