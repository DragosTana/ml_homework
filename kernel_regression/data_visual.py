import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import misc



def main():
    # Load data
    knn_mse = misc.load_csv('error_knn_1000_100.csv')
    reg_mse = misc.load_csv('error_reg_1000_100.csv')
    knn = [item for sublist in knn_mse for item in sublist]
    reg = [item for sublist in reg_mse for item in sublist]
    
    mean_knn = sum(knn)/len(knn)
    mean_reg = sum(reg)/len(reg)
    #kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    #kde.fit(knn[:, None])
    plt.hist(knn, bins = 15, alpha=0.5, color="red", label='KNN')
    plt.hist(reg, bins = 15, alpha=0.5, color = "blue", label='Regression')
    plt.axvline(x = mean_knn, color = "red")
    plt.axvline(x = mean_reg,  color = "blue")
    plt.legend(loc='upper right')
    plt.title("100 montecarlo simulations with 500 samples")
    plt.show()
    
def plot_func():
    x, y = misc.data_generating_process(dimensions = 1000, feature = 1)
    plt.plot(y, x)
    plt.show()

if __name__ == '__main__':
    plot_func()