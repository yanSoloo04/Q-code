import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer

import matplotlib.pyplot as plt


dev = qml.device('default.qubit', wires=3)


@qml.qnode(dev)
def qcircuit(params, x, y):
    """A variational quantum circuit representing the Universal classifier.

    Args:
        params (array[float]): array of parameters
        x (array[float]): single input vector
        y (array[float]): single output state density matrix

    Returns:
        float: fidelity between output state and input
    """
    for p in params:
        qml.Rot(x[0], x[1], x[2], wires=0)
        qml.Rot(x[3], x[4], x[5], wires=1)
        qml.Rot(x[6], x[7], 0, wires=2)
    return qml.expval(qml.Hermitian(y, wires=[0]))


def cost(params, x, y, state_labels=None):
    """Cost function to be minimized.

    Args:
        params (array[float]): array of parameters
        x (array[float]): 2-d array of input vectors
        y (array[float]): 1-d array of targets
        state_labels (array[float]): array of state representations for labels

    Returns:
        float: loss value to be minimized
    """
    
    loss = 0.0 # we initialize loss value at 0
    dm_labels = [density_matrix(s) for s in state_labels]
    for i in range(len(x)):
        # we take the parameters
        input_vector = x[i][:8]
        # we take the label
        target_label = int(x[i][8])  # last element is the label so its in {0,1}
        
        f = qcircuit(params, input_vector, dm_labels[target_label])
        
        #we sum the errors for each data point
        loss = loss + (1 - f) ** 2


    x = [
    [132.9765625, 39.25068965, -0.190663109, 0.71935469, 2.005852843, 14.53679908, 9.956426993, 122.1164601, 0],
    [96.546875,35.72920273,0.419386559,2.228976323,3.341137124,21.00789818,8.122209194,75.10664451,0],
    [117.4375,47.61068518,0.149848307,0.148917285,2.445652174,14.67882087,9.672969814,120.4693113,0],
    [135.0703125,47.13812543,0.046870269,0.012329002,2.537625418,14.13626077,8.706149281,107.1620941,0]
    ]