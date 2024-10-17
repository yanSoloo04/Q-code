from pennylane import numpy as np
import pennylane as qml
import math
from typing import Tuple
from pennylane.optimize import NesterovMomentumOptimizer
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from data import get_samples

##TO SEE...###
from qiskit.circuit.library import RealAmplitudes
from pennylane import from_qiskit
from qiskit.circuit import ParameterVector



def layer(layer_weights: NDArray):
    """
    This function creates the circuit of one layer of the ansatz for the vqc. We use CNOT to create intrication.
    Args:
    layer_weights (NDArray): the weights used for the Ry for this layer of the ansatz
    """
    nb_qubits =3
    for i in range(nb_qubits):
        qml.RY(layer_weights[i], wires=i)
    for i in range(nb_qubits):
        if i == nb_qubits-1:
            qml.CNOT([i, 0])
        else:
            qml.CNOT([i, i+1])

    

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit(weights: NDArray, x: NDArray, ansatz:str, embedding:str, rot:str = ''):
    """ 
    This function creates the circuit for one iterration of the quantum variationnal classifier and evaluates the expected value of PauliZ on the first qubit.
    
    Args: 
    weights (NDArray): a numpy array containing the weights to optimize for each layer of the ansatz
    x (NDArray): the numpy array containing the parameters for the current data being classified
    ansatz (str): the type of ansatz that is used. Can only be random or layer.
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.

    Returns: the expected value of PauliZ of the first qubit of the circuit.
    """
    #amplitude embedding
    if embedding == 'amplitude':
        nb_qubits = int(math.log2(len(x)))
        qml.AmplitudeEmbedding(features = x, wires = range(nb_qubits), normalize=True)
    
    #angle embedding
    elif embedding == 'angle':
        assert rot in ['X', 'Y', 'Z'], 'rot as to be either X, Y, or Z for the angle embedding'
        nb_qubits = len(x)
        qml.AngleEmbedding(features = x, wires = range(nb_qubits), rotation = rot)

    #Random layer ansatz
    if ansatz == 'random':
        qml.RandomLayers(weights, wires = range(nb_qubits))

    #Ry and CNOT ansatz
    elif ansatz == 'layer':
        for layer_weights in weights:
            layer(layer_weights)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights: NDArray, bias :NDArray, x: NDArray, ansatz:str, embedding:str, rot:str = ''):
    """
    This function takes the expected value of PauliZ on the first qubit and add it to the bias which is classical parameter.

    Args:
    weights (NDArray): a numpy array containing the weights to optimize for each layer of the ansatz
    x (NDArray): the numpy array containing the parameters for the current data being classified
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.
    ansatz (str): the type of ansatz that is used. Can only be random or layer.
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.

    Returns: the expected value of PauliZ on the first qubit of the variationnal circuit added to the bias

    """
    return circuit(weights, x, ansatz, embedding, rot) + bias 

def square_loss(labels :NDArray, predictions : NDArray):
    """
    Compute the square_loss between the labels and the predictions for the cost function

    Args:
    labels (NDArray): a numpy array containing the actual labels
    predictions (NDArray): a numpy array containing the predicted labels

    Returns: the square loss between the labels and the predicted labels
    """
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels : NDArray, predictions : NDArray):
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

def cost(weights: NDArray, bias:NDArray, X : NDArray, Y : NDArray, ansatz:str, embedding:str, rot:str = ''):
    """
    Compute the square loss of the predictions using the vqc with the actual labels

    Args:
    weights (NDArray): the weights used for the ansatz that will be updated in a numpy array
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.
    X (NDArray): the numpy array containing the parameters for all of the datas
    Y (NDArray): the nuoy array containing the actual labels
    ansatz (str): the type of ansatz that is used. Can only be random or layer.
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.

    Returns:
    the square loss between the predicted labels and the actual labels
    """
    predictions = [variational_classifier(weights, bias, x, ansatz, embedding, rot) for x in X]
    return square_loss(Y, predictions)

def validate_entries(embedding:str, ansatz:str, nb_datas:int, batch_size:int, num_layers:int, nb_iterations:int, labels_value:Tuple):
    assert embedding in ['angle', 'amplitude'], 'Two embedding methods are available: the angle embedding or the amplitude embedding'
    assert ansatz in ['random', 'layer'], 'two ansatz are available, the qml.randomlayers or a fixed one described with the function layer'
    assert nb_datas%2 == 0, 'the number of datas has to be divisible by two because we take 50/50 datas with the labels'
    assert batch_size%2==0, 'the number of data for a batch has to be divisible by two because we take 50/50 datas with the labels'
    assert len(labels_value) == 2, 'our VQC can only classify two different labels at a time. len(labels) as to be 2'
    assert num_layers>=2, 'the ansatz has to use 2 or more layers'
    assert nb_iterations >=1, 'the VQC has to run at least one time...'

def run_VQC(dataset:NDArray, nb_datas: int, batch_size:int, labels_value: Tuple, ansatz:str, embedding:str, rot:str = '', nb_iterations:int = 20, num_layers:int = 2)-> float:
    """
    This function takes a dataset and trains a VQC with the givens parameters. The VQC is tested at each optimisation step to see the accuracy changing. 

    Args:
    dataset (NDArray): the numpy array containing the dataset with the parameters and the corresponding labels. The last column of the dataset should be the labels associated with the parameters in the same line.
    nb_datas (int): the number of data used for testing at each iteration
    batch_size (int): the number of data used for training the VQC at each iteration
    labels_value (Tuple): the labels to be classified in the dataset. The labels must be present on the last column of the dataset at least nb_data/2. This argument should have this form: [label1, label2]
    ansatz (str): the type of ansatz that is used. Can only be random or layer.
    embedding (str): the type of embedding used to embedd the data. Can only be amplitude or angle
    rot (str): the axis of the rotation for the angle embedding. Can only be 'X', 'Y' or 'Z' or '' if the embedding choosen is amplitude.
    nb_iterations (int): the number of iterations to do for training the VQC
    num_layers (int): the number of layers to use for ansatz

    Returns: the accuracy score after all the iterations of optimisation for the classification of the data using the VQC method
    """
    #validating the entries with some asserts described in the function validate_entries
    validate_entries(embedding, ansatz, nb_datas, batch_size, num_layers, nb_iterations, labels_value)

    #we get the parameters and the labels associated with the parameters in two distinct arrays
    parameters, labels= get_samples(dataset, nb_datas, labels_value)

    #We take the max to be pi/2 and the rest of the parameters to be less than pi/2
    scaler = StandardScaler().fit(parameters)
    X = scaler.transform(parameters)
    #we choose a seed for the random to be comparable using different methods
    np.random.seed(0)

    num_qubits = len(X[0])

    #initialization of the bias and the weights which are random
    weights = np.random.randn(num_layers, num_qubits)
    bias = np.array(0.0)

    opt = NesterovMomentumOptimizer(0.35)

    #iteration to optimise the vqc for better results
    for it in range(nb_iterations):

        # Creating a batch to train our VQC on one iteration
        X_batch_to_reduce, Y_batch= get_samples(dataset, batch_size, labels_value)
        scaler = StandardScaler().fit(X_batch_to_reduce)
        X_batch = scaler.transform(X_batch_to_reduce)

        # Update the weights and the bias by one optimizer step
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch, ansatz = ansatz, embedding = embedding, rot = rot)

        # Compute predictions using np.sign for the labels to be -1 or 1
        predictions = [np.sign(variational_classifier(weights, bias, x, ansatz, embedding, rot)) for x in X]

        #Printing the cost and the accuracy of the current iteration
        current_cost = cost(weights, bias, X, labels, ansatz, embedding, rot)
        acc = accuracy(labels, predictions)
        print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
        print('-----------------------------------------------------------------------------------------')
        
    #Printing the labels for visual interpretation
    print('Actual labels: ', labels)
    print('Predicted labels: ', np.array(predictions))
    return acc
