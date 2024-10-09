from pennylane import numpy as np
import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
from numpy.typing import NDArray
from data import get_samples, get_csv_file, get_xlsx_file



def layer(layer_weights: NDArray):
    """
    This function creates the circuit of one layer of the ansatz for the vqc
    Args:
    layer_weights (NDArray): the weights for this layer of the ansatz
    """
    nb_qubits = len(layer_weights)
    for i in range(nb_qubits):
        print(layer_weights)
        qml.RY(layer_weights[i], wires=i)
    for i in range(nb_qubits):
        if i == nb_qubits-1:
            qml.CNOT([i, 0])
        qml.CNOT([i, i+1])

    

dev = qml.device("default.qubit")
@qml.qnode(dev)
def circuit(weights: NDArray, x: NDArray):
    """ 
    This function creates the circuit for one iterration of the quantum variationnal classifier and evaluates the expected value of PauliZ on the first qubit.
    
    Args: 
    weights (NDArray): a numpy array containing the weights to optimize for each layer of the ansatz
    x (NDArray): the numpy array containing the parameters for the current data being classified

    Returns: the expected value of PauliZ of the first qubit of the circuit.
    """
    nb_qubits = len(x)
    qml.AngleEmbedding(features = x, wires = range(nb_qubits))
    qml.RandomLayers(weights, wires = range(nb_qubits))
    # for layer_weights in weights:
    #     layer(layer_weights)
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

#we get the parameters and the labels associated with the parameters in two distinct arrays
nb_datas = 50
x = get_csv_file("HTRU_2.csv")
X_to_reduce, y= get_samples(x, nb_datas, [0, 1])

#We take the max to be pi/2 and the rest of the parameters to be less than pi/2
m = np.max(X_to_reduce)
X = X_to_reduce/m*np.pi/2



#we choose a seed for the random to be comparable using different methods
np.random.seed(0)

num_qubits = len(X[0])

#initialisation of the bias and the weights which are random
num_layers = 2
weights = 0.01 * np.random.randn(num_layers, num_qubits, requires_grad = True)
bias = np.array(0.0, requires_grad = True)

opt = NesterovMomentumOptimizer(0.52)

#iteration to optimise the vqc for better results
nb_iterations = 20
for it in range(nb_iterations):

    # Update the weights by one optimizer step
    weights, bias = opt.step(cost, weights, bias, X=X, Y=y)

    # Compute accuracy using np.sign for the labels to be -1 or 1
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

    #Printing the cost and the accuracy of the current iteration
    current_cost = cost(weights, bias, X, y)
    acc = accuracy(y, predictions)
    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
    #Printing the labels for the person behind the chair hehe
    print('Actual labels: ', y)
    print('Predicted labels: ', np.array(predictions))
    print('-----------------------------------------------------------------------------------------')
