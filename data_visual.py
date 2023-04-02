import matplotlib.pyplot as plt
import csv

def load_csv(file_name):
    data = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            point = []
            for word in line:
                point.append(float(word))
            data.append(point)
    return data

def main():
    # Load data
    knn_mse = load_csv('error_knn_1000.csv')
    reg_mse = load_csv('error_reg_1000.csv')
    knn = [item for sublist in knn_mse for item in sublist]
    reg = [item for sublist in reg_mse for item in sublist]
    
    plt.hist(knn, bins = 20, alpha=0.5, color="red", label='KNN')
    plt.hist(reg, bins = 20, alpha=0.5, color = "blue", label='Regression')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == '__main__':
    main()