import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib import cm
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


dev = qml.device('default.qubit', wires=3)

# feature: data that we wish to embed
# wires: wires on which we wish to embed the data
# normalize: if we wish for the state to be automatically normalized

@qml.qnode(dev)
def amplitude_circuit(f=None):
    qml.AmplitudeEmbedding(features=f, wires=range(3), normalize=True)
    return qml.expval(qml.Z(0)), qml.state()

# example with data point numbered 15428 -> 0
data_point = [135.0703125,47.13812543,0.046870269,0.012329002,2.537625418,14.13626077,8.706149281,107.1620941]
result, state = amplitude_circuit(f=data_point)
print(state)

print("Expectation value:", result)
print("Quantum state:", state)

