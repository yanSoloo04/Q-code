from pennylane import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.preprocessing import StandardScaler
from data import get_samples
import math
from typing import Tuple


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
def qcnn_circuit(parameters:NDArray, weights: NDArray, embedding: str, rot: str = ''):
    """
    Creates the circuit for the qcnn using the parameters of one data
    Args:
    parameters (NDArray): the parameters for the data that we want to use
    weights (NDArray): trainable parameters for the QCNN to learn and get better results
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.

    Returns: The expected value of the first qubit after applying the QCNN circuit
    """
    

    ##Choosing the embedding
    if embedding == 'amplitude':
        nb_qubits_init = int(math.log2(len(parameters)))
        qml.AmplitudeEmbedding(parameters, wires = range(nb_qubits_init), normalize=True)

    elif embedding == 'angle':
        assert rot in ['X', 'Y', 'Z'], 'rot must be X, Y or Z for the angle embedding'
        nb_qubits_init = len(parameters)
        qml.AngleEmbedding(features= parameters, wires = range(nb_qubits_init), rotation=rot)

    
    nb_qubits = nb_qubits_init
    while nb_qubits != 1:
        # one step of convolution and pooling
        convolution_circuit(weights[:nb_qubits])
        # qml.RandomLayers(weights, wires = range(nb_qubits))
        qml.Barrier(wires = range(nb_qubits_init))
        pooling_circuit(nb_qubits)
        qml.Barrier(wires = range(nb_qubits_init))
        #If the number of qubits is odd, we use math.ceil because there's one qubit not used in the pooling phase
        nb_qubits = math.ceil(nb_qubits/2)
    return qml.expval(qml.PauliZ(0))

def qcnn_classifier(parameters, weights, bias, embedding: str, rot: str = ''):
    """
    This function takes the expected value of PauliZ on the first qubit and add it to the bias which is classical parameter.

    Args:
    weights (NDArray): a numpy array containing the weights to optimize for the convolutionnal step
    x (NDArray): the numpy array containing the parameters for the current data being classified
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.

    Returns: the expected value of PauliZ on the first qubit of the QCNN circuit added to the bias
    """
    return qcnn_circuit(parameters, weights, embedding, rot) + bias

def cost(weights, bias, X, Y, embedding: str, rot: str = ''):
    """
    Compute the square loss of the predictions using the QCNN with the actual labels

    Args:
    weights (NDArray): the weights used for the convolution step that will be updated in a numpy array
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.
    X (NDArray): the numpy array containing the parameters for all of the datas
    Y (NDArray): the nuoy array containing the actual labels
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.

    Returns:
    the square loss between the predicted labels and the actual labels
    """
    predictions = [qcnn_classifier(x, weights, bias, embedding, rot) for x in X]
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

def validate_entries(nb_datas: int, batch_size: int, labels_value: Tuple, embedding: str, nb_iterations: int):
    assert embedding in ['angle', 'amplitude'], 'Two embedding methods are available: the angle embedding or the amplitude embedding'
    assert nb_datas%2 == 0, 'the number of datas has to be divisible by two because we take 50/50 datas with the labels'
    assert batch_size%2==0, 'the number of data for a batch has to be divisible by two because we take 50/50 datas with the labels'
    assert len(labels_value) == 2, 'our QCNN can only classify two different labels at a time. len(labels) as to be 2'
    assert nb_iterations >=1, 'the QCNN has to run at least one time...'


def run_QCNN(dataset: NDArray, nb_datas: int, batch_size: int, labels_value: Tuple, embedding: str, rot: str = '', nb_iterations: int = 20):
    """
    This function takes a dataset and trains a QCNN with the givens parameters. The QCNN is tested at each optimisation step to see the accuracy changing. 

    Args:
    dataset (NDArray): the numpy array containing the dataset with the parameters and the corresponding labels. The last column of the dataset should be the labels associated with the parameters in the same line.
    nb_datas (int): the number of data used for testing at each iteration
    batch_size (int): the number of data used for training the QCNN at each iteration
    labels_value (Tuple): the labels to be classified in the dataset. The labels must be present on the last column of the dataset at least nb_data/2. This argument should have this form: [label1, label2]
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.
    nb_iterations (int): the number of iterations to do for training the QCNN
    num_layers (int): the number of layers to use for ansatz

    Returns: the accuracy score after all the iterations of optimisation for the classification of the data using the QCNN method
    """

    validate_entries(nb_datas, batch_size, labels_value, embedding, nb_iterations)

    parameters, labels = get_samples(dataset, nb_datas, labels_value)

    #scaling the parameters
    scaler = StandardScaler().fit(parameters)
    X = scaler.transform(parameters)

    #setting a seed for the weights for comparison between different embeddings
    np.random.seed(0)

    nb_qubits = len(X[0])

    #initialization of the bias and the weights which are random
    weights = 0.07 * np.random.randn(nb_qubits)
    bias = np.array(0.0)

    opt = NesterovMomentumOptimizer(0.5)

    #iteration to optimise the qcnn for better results
    for it in range(nb_iterations):

        #setting the training batch for one iteration 
        X_batch_to_reduce, Y_batch= get_samples(dataset, batch_size, labels_value)
        scaler = StandardScaler().fit(X_batch_to_reduce)
        X_batch = scaler.transform(X_batch_to_reduce)

        # Update the weights by one optimizer step using a training batch
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch, embedding = embedding, rot = rot)

        # Compute predictions using np.sign for the labels to be -1 or 1
        predictions = [np.sign(qcnn_classifier(x, weights, bias, embedding, rot)) for x in X]

        #Printing the cost and the accuracy of the current iteration
        current_cost = cost(weights, bias, X, labels, embedding, rot)
        acc = accuracy(labels, predictions)
        print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
        print('-----------------------------------------------------------------------------------------')

    #Printing the labels for visual interpretation
    print('Actual labels: ', labels)
    print('Predicted labels: ', np.array(predictions))
    return acc