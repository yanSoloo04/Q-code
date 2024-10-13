from pennylane import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane.optimize import NesterovMomentumOptimizer
from data import get_csv_file, get_samples, get_xlsx_file
import math


def pooling_circuit(nb_qubits: int):
    """
    This function creates the circuit to confine the information in n/2 qubits.

    Args:
    nb_qubits (int): the number of qubits used to create the circuit
    """
    m = math.floor(nb_qubits/2)
    for i in range(m):
        j = nb_qubits-1-i
        qml.CNOT(wires = [j, i])
    

def convolution_circuit(weights:NDArray):
    """
    This function creates the circuit representing a convolution step in the QCNN.

    Args:
    weights (NDArray): the parameters to use in the rotation gates for training.
    """
    nb_qubits = len(weights)
    for i in range(nb_qubits):
        j = (i+5)%nb_qubits
        qml.RY(phi = weights[i], wires = i)
        qml.RY(phi = weights[j], wires = j)
        qml.CNOT(wires = [i, j])


dev = qml.device("default.qubit")
@qml.qnode(dev)
def qcnn_circuit(parameters:NDArray, weights: NDArray):
    """
    Creates the circuit for the qcnn using the parameters of one data
    Args:
    parameters (NDArray): the parameters for the data that we want to use
    weights (NDArray): trainable parameters for the QCNN to learn and get better results

    Returns: The expected value of the first qubit after applying the QCNN circuit
    """

    nb_qubits_init = 4
    qml.AmplitudeEmbedding(parameters, wires = range(nb_qubits_init), normalize=True)
    nb_qubits = nb_qubits_init
    while nb_qubits != 1:
        convolution_circuit(weights[:nb_qubits])
        # qml.RandomLayers(weights, wires = range(nb_qubits))
        qml.Barrier(wires = range(nb_qubits_init))
        pooling_circuit(nb_qubits)
        qml.Barrier(wires = range(nb_qubits_init))
        nb_qubits = math.ceil(nb_qubits/2)
    return qml.expval(qml.PauliZ(0))

def qcnn_classifier(parameters, weights, bias):
    """
    This function takes the expected value of PauliZ on the first qubit and add it to the bias which is classical parameter.

    Args:
    weights (NDArray): a numpy array containing the weights to optimize for the convolutionnal step
    x (NDArray): the numpy array containing the parameters for the current data being classified
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.

    Returns: the expected value of PauliZ on the first qubit of the QCNN circuit added to the bias
    """
    return qcnn_circuit(parameters, weights) + bias

def cost(weights, bias, X, Y):
    """
    Compute the square loss of the predictions using the QCNN with the actual labels

    Args:
    weights (NDArray): the weights used for the convolution step that will be updated in a numpy array
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.
    X (NDArray): the numpy array containing the parameters for all of the datas
    Y (NDArray): the nuoy array containing the actual labels

    Returns:
    the square loss between the predicted labels and the actual labels
    """
    predictions = [qcnn_classifier(x, weights, bias) for x in X]
    return square_loss(Y, predictions)

def accuracy(labels, predictions):
    """
    Compute the accuracy of the prediction using the labels of the parameters

    Args:
    labels (NDArray): a numpy array containing the actual labels
    predictions (NDArray): a numpy array containing the predicted labels

    Returns: the accuracy of the predictions
    """
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def square_loss(labels, predictions):
    """
    Compute the square_loss between the labels and the predictions for the cost function

    Args:
    labels (NDArray): a numpy array containing the actual labels
    predictions (NDArray): a numpy array containing the predicted labels

    Returns: the square loss between the labels and the predicted labels
    """
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

#getting the dataset into an array
x = get_xlsx_file('Dry_Bean_Dataset.xlsx')
x_to_scale, y = get_samples(x, 50, ['SIRA', 'DERMASON'])

#scaling the parameters
m = np.max(x_to_scale)
X = x_to_scale/m*np.pi/2

#setting a seed for the weights for comparison between different embeddings
np.random.seed(0)

nb_qubits = len(X[0])

#initialization of the bias and the weights which are random
# weights = 0.07 * np.random.randn(nb_qubits)
weights = np.array([-0.15155443,  0.03289792, -0.14296978,  0.01073419, -0.02191593, -0.0019281,
 -0.14784011, -0.0409323,  -0.00325512,  0.02059717, -0.11453522,  0.06808275,
 -0.03777734, -0.09488475,  0.01733188,  0.05791952])
bias = np.array(0.0)
norm = np.linalg.norm(weights)
weights = weights/norm

opt = NesterovMomentumOptimizer(0.15)
batch_size = 20

#iteration to optimise the qcnn for better results
nb_iterations = 40
for it in range(nb_iterations):

    # Update the weights by one optimizer step
    X_batch_to_reduce, Y_batch= get_samples(x, batch_size, ['SIRA', 'DERMASON'])
    m = np.max(X_batch_to_reduce)
    X_batch = X_batch_to_reduce/m*np.pi/2
    weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

    # Compute accuracy using np.sign for the labels to be -1 or 1
    predictions = [np.sign(qcnn_classifier(x, weights, bias)) for x in X]

    #Printing the cost and the accuracy of the current iteration
    current_cost = cost(weights, bias, X, y)
    acc = accuracy(y, predictions)
    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
    #Printing the labels for visual interpretation
    print('Actual labels: ', y)
    print('Predicted labels: ', np.array(predictions))
    print('-----------------------------------------------------------------------------------------')

