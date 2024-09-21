import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np 

import pandas as pd
from matplotlib import cm

dev = qml.device('default.qubit', wires=3)

# feature: data that we wish to embed
# wires: wires on which we wish to embed the data
# normalize: if we wish for the state to be automatically normalized

@qml.qnode(dev)
def amplitude_circuit(f=None):
    qml.AmplitudeEmbedding(features=f, wires=range(3), normalize=True)
    return qml.expval(qml.PauliZ(0)), qml.state()


# Load the dataset
data = pd.read_csv('HTRU_2.csv')

# select the 10000th row as an example
data_point = data.iloc[10000, :8]

# Convert the data point to a Nparray
data_point_np = data_point.to_numpy()

#print(data_point_np.shape)  # Should be (8,)
#print(data_point_np)  # Print to see the data point
#print(data_point_np.ndim == 1)  # Should return True


result, state = amplitude_circuit(f=data_point_np)
print(state)

print("Expectation value:", result)
print("Quantum state:", state)

result, state = amplitude_circuit(f=data_point_np)

# Output the results
print("Expectation value:", result)
print("Quantum state:", state)
