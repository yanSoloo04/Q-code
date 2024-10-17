import numpy as np
import pennylane as qml
from pennylane import AngleEmbedding, AmplitudeEmbedding
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from data import get_csv_file, get_samples, get_xlsx_file


device = qml.device("default.qubit")
@qml.qnode(device)
def kernel(x: NDArray, y: NDArray, embedding:str, rot:str = ''):
    """
    Evaluates the circuit that represents the kernel
    Args:
    x (NDArray): the parameters that represent the first data
    y (NDArray): the parameters that represent the second data
    embedding (str): the embedding to use for encoding the parameters. This argument can only be 'amplitude'or 'angle'
    rot (str): the axis of the rotation for the angle embedding (is '' if the embedding is amplitude). As to be either 'X', 'Y' or 'Z'.

    Returns: the probability of mesuring for all of the basis states given the state of the system.
    """
    assert(len(x) == len(y)), 'the parameters of the two datas must be of the same length'
    assert(embedding in ['amplitude', 'angle']), 'two methods are available for the embedding, amplitude or angle.'
    
    
    #amplitude embedding
    if embedding == 'amplitude':
        nb_qubits = int(math.log2(len(x)))
        AmplitudeEmbedding(features = x, wires = range(nb_qubits), normalize = True)
        qml.adjoint(AmplitudeEmbedding(normalize=True, features = y, wires = range(nb_qubits)))

    ##Angle embedding
    if embedding == 'angle':
        assert rot in ['X', 'Y', 'Z'], 'rot as to be either Y, X or Z for the angle embedding'
        nb_qubits = len(x)
        AngleEmbedding(features = x, wires = range(nb_qubits), rotation = rot)
        qml.adjoint(AngleEmbedding(features = y, wires = range(nb_qubits), rotation = rot))
    return qml.probs(wires = range(nb_qubits))


def kernel_matrix(A: NDArray, B: NDArray, embedding:str, rot:str)-> NDArray:
    """Compute the matrix whose entries are the kernel
       evaluated on pairwise data from sets A and B.
    """
    return np.array([[kernel(a, b, embedding, rot)[0] for b in B] for a in A])


def draw_kernel_matrix(matrix: NDArray, cmap: str, filename:str):
    """
    Draws the kernel matrix
    Args:
    matrix (NDArray): the matrix to plot
    cmap (str): the cmap used for the plot of the matrix
    filnema (str): the name of the file where the kernel matrix will be saved
    """
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, cmap = cmap)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amount of Classifications", rotation=270, labelpad=15, fontsize = 14)
    plt.plot()
    plt.savefig(filename)


def run_QSVM(parameters:NDArray, labels:NDArray, embedding:str, rot:str, filename:str)-> float:
    #scaling the parameters and setting a batch for training and one for testing
    scaler = StandardScaler().fit(parameters)
    X_scaled = scaler.transform(parameters)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)

    #Drawing the kernel matrix
    draw_kernel_matrix(kernel_matrix(X_train[1:50, :], X_train[1:50, :], embedding = embedding, rot = rot), cmap = 'binary', filename = filename)

    #predicting the labels using the quantum kernel matrix
    svm = SVC(kernel=lambda X, Y: kernel_matrix(X, Y, embedding=embedding, rot=rot)).fit(X_train, y_train)
    predictions = svm.predict(X_test)

    #printing the labels for comparison
    print(predictions, y_test)

    #returning the accuracy score
    acc = accuracy_score(predictions, y_test)
    return acc


#getting the parameters and their labels from the file
x = get_csv_file('HTRU_2.csv')
X, y= get_samples(x, 100, [0, 1])
acc = run_QSVM(X, y, 'amplitude', '')
print(acc)