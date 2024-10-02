from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from pennylane.optimize import NesterovMomentumOptimizer
import pandas as pd
import random

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




def pooling_circuit(nb_qubits: int):
    assert(nb_qubits%2 == 0)
    m = int(nb_qubits/2)
    for i in range(m):
        j = nb_qubits-1-i
        qml.CNOT(wires = [j, i])
    

def convolution_circuit(weights:NDArray):
    nb_qubits = len(weights)
    for i in range(nb_qubits):
        j = (i+5)%nb_qubits
        qml.RY(phi = weights[i], wires = i)
        qml.RY(phi = weights[j], wires = j)
        qml.CNOT(wires = [i, j])
    


dev = qml.device("default.qubit")
@qml.qnode(dev)
def qcnn_circuit(parameters:NDArray, weights: NDArray):
    nb_qubits_init = len(parameters)
    qml.AngleEmbedding(parameters, wires = range(nb_qubits_init))
    nb_qubits = nb_qubits_init
    while nb_qubits%2 == 0:
        convolution_circuit(weights[:nb_qubits])
        qml.Barrier(wires = range(nb_qubits_init))
        pooling_circuit(nb_qubits)
        qml.Barrier(wires = range(nb_qubits_init))
        nb_qubits = int(nb_qubits/2)
    return qml.expval(qml.PauliZ(0))

def qcnn_classifier(parameters, weights, bias):
    return qcnn_circuit(parameters, weights) + bias

def cost(weights, bias, X, Y):
    predictions = [qcnn_classifier(weights, x, bias) for x in X]
    return square_loss(Y, predictions)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

    
x = get_data_file()
X, y= get_samples(x, 50)


m = np.max(X)
X = X/m*np.pi/2

y = y*2-1


np.random.seed(0)

nb_qubits = len(X[0])
weights = 0.99 * np.random.randn(nb_qubits)
bias = np.array(0.0)

opt = NesterovMomentumOptimizer(0.5)

for it in range(100):

    # Update the weights by one optimizer step
    weights, bias = opt.step(cost, weights, bias, X=X, Y=y)

    # Compute accuracy
    predictions = [np.sign(qcnn_classifier(x, weights, bias)) for x in X]

    current_cost = cost(weights, bias, X, y)
    acc = accuracy(y, predictions)
    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")

