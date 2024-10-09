from pennylane import numpy as np
import pennylane as qml
from numpy.typing import NDArray
from pennylane.optimize import NesterovMomentumOptimizer
from data import get_csv_file, get_samples, get_xlsx_file


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
        # qml.RandomLayers(weights, wires = range(nb_qubits))
        qml.Barrier(wires = range(nb_qubits_init))
        pooling_circuit(nb_qubits)
        qml.Barrier(wires = range(nb_qubits_init))
        nb_qubits = int(nb_qubits/2)
    return qml.expval(qml.PauliZ(0))

def qcnn_classifier(parameters, weights, bias):
    return qcnn_circuit(parameters, weights) + bias

def cost(weights, bias, X, Y):
    predictions = [qcnn_classifier(x, weights, bias) for x in X]
    return square_loss(Y, predictions)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def square_loss(labels, predictions):
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

    
x = get_xlsx_file('Dry_Bean_Dataset.xlsx')
x_to_scale, y = get_samples(x, 50, ['SIRA', 'DERMASON'])


m = np.max(x_to_scale)
X = x_to_scale/m*np.pi/2

#setting a seed for the weights for comparison between different embeddings
np.random.seed(69420)

nb_qubits = len(X[0])
# weights = 0.07 * np.random.randn(nb_qubits)
weights = np.array([-0.15155443,  0.03289792, -0.14296978,  0.01073419, -0.02191593, -0.0019281,
 -0.14784011, -0.0409323,  -0.00325512,  0.02059717, -0.11453522,  0.06808275,
 -0.03777734, -0.09488475,  0.01733188,  0.05791952])
bias = np.array(0.0)

opt = NesterovMomentumOptimizer(0.19)

nb_iterations = 20
for it in range(nb_iterations):

    # Update the weights by one optimizer step
    weights, bias = opt.step(cost, weights, bias, X=X, Y=y)

    # Compute accuracy
    predictions = [np.sign(qcnn_classifier(x, weights, bias)) for x in X]

    current_cost = cost(weights, bias, X, y)
    acc = accuracy(y, predictions)
    print(f"Iter: {it+1:4d} | Cost: {current_cost:0.7f} | Accuracy: {acc:0.7f}")
    #Printing the labels for the person behind the chair hehe
    print('Actual labels: ', y)
    print('Predicted labels: ', np.array(predictions))
    print('-----------------------------------------------------------------------------------------')
    

