from pennylane import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from pennylane.optimize import NesterovMomentumOptimizer
from numpy.typing import NDArray
import random
import pandas as pd
from scipy.optimize import minimize


def get_htru_2_datas(filename: str)->tuple[NDArray, NDArray]:
    data = np.genfromtxt(filename, delimiter=',', skip_header=0)  
    X = data[:, :-1]
    Y = data[:, -1]
    return X, Y
def get_data_file():

    data_file = pd.read_csv('HTRU_2.csv')
    data_array = data_file.to_numpy()

    return data_array



def layer(layer_weights):
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
def circuit(weights, x):
    nb_qubits = len(x)
    qml.AngleEmbedding(features = x, wires = range(nb_qubits))
    
    qml.RandomLayers(weights, wires = range(nb_qubits))
    return qml.expval(qml.PauliZ(0))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def cost(weights, bias, X, Y):
    predictions = [variational_classifier(weights, bias, x) for x in X]
    return square_loss(Y, predictions)

def get_samples(data, sample_amount):

    sample_array = []
    zero_value = 1
    one_value = 1
    single_data = []

    while len(sample_array) < sample_amount:
        single_data = random.choice(data)
        
        if single_data[8] == 0.0 and zero_value <= (sample_amount/2):
            sample_array.append(single_data)
            zero_value += 1

        elif single_data[8] == 1.0 and one_value <= (sample_amount/2):
            sample_array.append(single_data)
            one_value += 1
  
    return sample_array

# X, y = get_htru_2_datas('HTRU_2.csv')
x = get_data_file()
x= np.array(get_samples(x, 50))
y = x[:, -1]
X = x[:, :-1]
m = np.max(X)
X = X/m*np.pi/2

y = y*2-1

np.random.seed(0)
num_qubits = 8
num_layers = 2
weights_init = 0.01 * np.random.randn(num_layers, num_qubits)
bias_init = np.array(0.0)

opt = NesterovMomentumOptimizer(0.52)

weights = weights_init
bias = bias_init
for it in range(100):

    # Update the weights by one optimizer step, using only a limited batch of data
    batch_index = np.random.randint(0, len(X), (50,))
    X_batch = X[batch_index]
    Y_batch = y[batch_index]
    weights, bias = opt.step(cost, weights, bias, X=X, Y=y)

    # Compute accuracy
    predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X]

    current_cost = cost(weights, bias, X, y)
    acc = accuracy(y, predictions)
    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
