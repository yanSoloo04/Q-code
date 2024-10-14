import numpy as np
import pennylane as qml
from pennylane import AngleEmbedding, AmplitudeEmbedding
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import get_csv_file, get_samples, get_xlsx_file

def parameters_in_angles(x_train: NDArray)-> NDArray:
    """
    This function takes parameters and put scale them between 0 and pi/2.

    Args:
    x_train (NDArray): an array containing all the parameters of the data we want to scale in multiple layers

    Returns:
    The scaled array
    """
    for i in range(len(x_train)):
        m = np.max(x_train[i])
        x_train[i] = x_train[i]/m*np.pi/2
    return x_train


device = qml.device("default.qubit")
@qml.qnode(device)
def kernel(x: NDArray, y: NDArray):
    """
    Evaluates the circuit that represents the kernel
    Args:
    x (NDArray): the parameters that represent the first data
    y (NDArray): the parameters that represent the second data

    Returns: the probability of mesuring for all of the basis states given the state of the system.
    """
    assert(len(x) == len(y))

    ##amplitude embedding
    # nb_qubits = 3
    # AmplitudeEmbedding(features = x, wires = range(nb_qubits), normalize = True)
    # qml.adjoint(AmplitudeEmbedding(normalize=True, features = y, wires = range(nb_qubits)))

    ##Angle embedding
    nb_qubits = len(x)
    AngleEmbedding(features = x, wires = range(nb_qubits), rotation = 'Y')
    qml.adjoint(AngleEmbedding(features = y, wires = range(nb_qubits), rotation = 'Y'))
    return qml.probs(wires = range(nb_qubits))


def kernel_matrix(A: NDArray, B: NDArray)-> NDArray:
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B.
    """
    return np.array([[kernel(a, b)[0] for b in B] for a in A])


def draw_kernel_matrix(matrix: NDArray, cmap: str):
    """
    Draws the kernel matrix
    Args:
    matrix (NDArray): the matrix to plot
    cmap (str): the cmap used for the plot of the matrix
    """
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
