import numpy as np
import pennylane as qml
from pennylane import AngleEmbedding
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import pandas as pd


def parameters_in_angles(x_train: NDArray)-> NDArray:
    for i in range(len(x_train)):
        m = np.max(x_train[i])
        x_train[i] = x_train[i]/m*np.pi/2

    return x_train


device = qml.device("default.qubit")
@qml.qnode(device)
def kernel(x: NDArray, y: NDArray):
    assert(len(x) == len(y))
    nb_qubits = len(x)
    

    AngleEmbedding(features = x, wires = range(nb_qubits))
    qml.adjoint(AngleEmbedding(features = y, wires = range(nb_qubits)))
    return qml.probs(wires = range(8))


def kernel_matrix(A: NDArray, B: NDArray)-> NDArray:
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B."""
    return np.array([[kernel(a, b)[0] for b in B] for a in A])


def draw_kernel_matrix(matrix: NDArray, cmap: str):
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap = cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amount of Classifications", rotation=270, labelpad=15, fontsize = 14)
    plt.plot()


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


draw_kernel_matrix(kernel_matrix(X_train[1:50, :], X_train[1:50, :]), cmap = 'BuGn')

svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)

predictions = svm.predict(X_test)

print(predictions, y_test)

print(accuracy_score(predictions, y_test))
