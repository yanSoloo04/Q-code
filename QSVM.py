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
from data import get_csv_file, get_samples, get_xlsx_file


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

#getting the parameters and their labels from the file
x = get_csv_file('HTRU_2.csv')
X, y= get_samples(x, 50, [0, 1])

#scaling the parameters
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
draw_kernel_matrix(kernel_matrix(X_train[1:50, :], X_train[1:50, :]), cmap = 'BuGn')

#predicting the labels using the quantum kernel matrix
svm = SVC(kernel=kernel_matrix).fit(X_train, y_train)
predictions = svm.predict(X_test)

#printing the accuracy score and the labels for comparison
print(predictions, y_test)
print(accuracy_score(predictions, y_test))
