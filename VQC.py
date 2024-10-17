from pennylane import numpy as np
import pennylane as qml
import math
from pennylane.optimize import NesterovMomentumOptimizer
from numpy.typing import NDArray
from data import get_samples, get_csv_file, get_xlsx_file
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

    Returns: the expected value of PauliZ of the first qubit of the circuit.
    """
    #amplitude embedding
    if embedding == 'amplitude':
        nb_qubits = math.log2(len(x))
        qml.AmplitudeEmbedding(features = x, wires = range(nb_qubits), normalize=True)
    
    #angle embedding
    if embedding == 'angle':
        assert rot in ['X', 'Y', 'Z'], 'rot as to be either X, Y, or Z for the angle embedding'
        nb_qubits = len(x)
        qml.AngleEmbedding(features = x, wires = range(nb_qubits), rotation = rot)

    #Random layer ansatz
    if ansatz == 'random':
        qml.RandomLayers(weights, wires = range(nb_qubits))

    #Ry and CNOT ansatz
    if ansatz == 'layer':
        for layer_weights in weights:
            layer(layer_weights)

    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights: NDArray, bias :NDArray, x: NDArray):
    """
    This function takes the expected value of PauliZ on the first qubit and add it to the bias which is classical parameter.

    Args:
    weights (NDArray): a numpy array containing the weights to optimize for each layer of the ansatz
    x (NDArray): the numpy array containing the parameters for the current data being classified
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.

    Returns: the expected value of PauliZ on the first qubit of the variationnal circuit added to the bias

    """
    return circuit(weights, x) + bias 

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

def cost(weights: NDArray, bias:NDArray, X : NDArray, Y : NDArray):
    """
    Compute the square loss of the predictions using the vqc with the actual labels

    Args:
    weights (NDArray): the weights used for the ansatz that will be updated in a numpy array
    bias (NDArray): a numpy array containing only the bias which is one of the variables that isn't fed into the gates of the variational circuit.
    X (NDArray): the numpy array containing the parameters for all of the datas
    Y (NDArray): the nuoy array containing the actual labels

    Returns:
    the square loss between the predicted labels and the actual labels
    """
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)


def run_VQC(dataset:NDArray, nb_datas: int, batch_size:int, labels: tuple, ansatz:str, embedding:str, rot:str = '', nb_iterations:int = 20, num_layers:int = 2)-> float:
    assert embedding in ['angle', 'amplitude'], 'Two embedding methods are available: the angle embedding or the amplitude embedding'
    assert ansatz in ['random', 'layer'], 'two ansatz are available, the qml.randomlayers or a fixed one described with the function layer'
    assert nb_datas%2 == 0, 'the number of datas has to be divisible by two because we take 50/50 datas with the labels'
    assert batch_size%2==0, 'the number of data for a batch has to be divisible by two because we take 50/50 datas with the labels'
    assert len(labels) == 2, 'our VQC can only classify two different labels at a time. len(labels) as to be 2'
    assert num_layers>=2, 'the ansatz has to use 2 or more layers'
    assert nb_iterations >=1, 'the VQC has to run at least one time...'


    #we get the parameters and the labels associated with the parameters in two distinct arrays
    X_to_reduce, y= get_samples(dataset, nb_datas, labels)

    #We take the max to be pi/2 and the rest of the parameters to be less than pi/2
    m = np.max(X_to_reduce)
    X = X_to_reduce/m*np.pi/2
    #we choose a seed for the random to be comparable using different methods
    np.random.seed(0)

    num_qubits = len(X[0])

    #initialization of the bias and the weights which are random
    weights = np.random.randn(num_layers, num_qubits, requires_grad = True)
    bias = np.array(0.0)

    # weights = np.array([[-0.15155443,  0.03289792, -0.14296978,  0.01073419, -0.02191593, -0.0019281,
    # -0.14784011, -0.0409323,  -0.00325512,  0.02059717, -0.11453522,  0.06808275,
    # -0.03777734, -0.09488475,  0.01733188,  0.05791952], [-0.15155443,  0.03289792, -0.14296978,  0.01073419, -0.02191593, -0.0019281,
    # -0.14784011, -0.0409323,  -0.00325512,  0.02059717, -0.11453522,  0.06808275,
    # -0.03777734, -0.09488475,  0.01733188,  0.05791952]])

    opt = NesterovMomentumOptimizer(0.35)

    #iteration to optimise the vqc for better results
    for it in range(nb_iterations):

        # Update the weights by one optimizer step
        X_batch_to_reduce, Y_batch= get_samples(x, batch_size, ['SIRA', 'DERMASON'])
        m = np.max(X_batch_to_reduce)
        X_batch = X_batch_to_reduce/m*np.pi/2
        weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

        # Compute predictions using np.sign for the labels to be -1 or 1
        predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

        #Printing the cost and the accuracy of the current iteration
        current_cost = cost(weights, bias, X, y)
        acc = accuracy(y, predictions)
        print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
        #Printing the labels for visual interpretation
        print('Actual labels: ', y)
        print('Predicted labels: ', np.array(predictions))
        print('-----------------------------------------------------------------------------------------')

    return acc

nb_datas = 50
x = get_xlsx_file("Dry_Bean_Dataset.xlsx")