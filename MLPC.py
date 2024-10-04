import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd

def get_htru_2_datas(filename: str)->tuple[NDArray, NDArray]:
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)  
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y

def get_data_file():

    data_file = pd.read_csv('HTRU_2.csv')
    data_array = data_file.to_numpy()

    return data_array

def get_samples(data, sample_amount):

    sample_array = []
    index_array = []
    zero_value = 1
    one_value = 1
    single_data = []

    while len(sample_array) < sample_amount:
        single_data = random.choice(data)
        
        if single_data[8] == 0.0 and zero_value <= (sample_amount/2):
            single_index = single_data[-1]
            single_data = single_data[:8]
            sample_array.append(single_data)
            index_array.append(single_index)
            zero_value += 1

        elif single_data[8] == 1.0 and one_value <= (sample_amount/2):
            single_index = single_data[-1]
            single_data = single_data[:8]
            sample_array.append(single_data)
            index_array.append(single_index)
            one_value += 1
    
    return np.array(sample_array), np.array(index_array, dtype = int)

x = get_data_file()
X, y= get_samples(x, 50)

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

y_scaled = 2 * y -1

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

mlp = MLPClassifier(hidden_layer_sizes=(100), max_iter=20000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

print(predictions, y_test)
print("Accuracy:", accuracy_score(y_test, predictions))

def draw_confusion_matrix(matrix: NDArray, title: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='coolwarm', aspect='equal')
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

cm = confusion_matrix(y_test, predictions)
draw_confusion_matrix(cm, "Confusion Matrix")