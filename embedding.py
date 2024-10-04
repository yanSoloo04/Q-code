import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np 

import pandas as pd
from matplotlib import cm

dev1 = qml.device('default.qubit', wires=3)
dev2 = qml.device('default.qubit', wires=8)
dev3 = qml.device('default.qubit', wires=8)

# feature: data that we wish to embed
# wires: wires on which we wish to embed the data
# normalize: if we wish for the state to be automatically normalized

@qml.qnode(dev1)
def amplitude_circuit(f=None):
    """Function that embeds data in a quantum circuit using amplitude embedding.

    Args:
        f = feature containing the data we wish to embed

    Returns:
        expectationMP: expectation value of the quantum state obtained (will be changed)
        quantum state: the quantum state obtained after the embedding
    """
    qml.AmplitudeEmbedding(features=f, wires=range(3), normalize=True)
    return qml.expval(qml.PauliZ(0)), qml.state()

@qml.qnode(dev2)
def basis_embedding(data, n_qubits):
    """Function that embeds data in a quantum circuit using basis embedding.

    Args:
        data = array containing the data we wish to embed

    Returns:
        expectation values: expectation value of the quantum state obtained (will be changed)
        quantum state: the quantum state obtained after the embedding
    """

    binary_array = data_to_binary_representation(data)
    qml.BasisState(binary_array, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0)), qml.state()

@qml.qnode(dev3)
def angle_embedding(x, y, n_qubits):

    assert(len(x) == len(y))
    n_qubits = len(x)

    qml.AngleEmbedding(features=x, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding(features = y, wires = range(n_qubits)))
    return qml.probs(wires = range(n_qubits))

def data_to_binary_representation(data):
    if (np.min(data) != np.max(data)):
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    else :
        raise ValueError("Input data is constant and can't be normalized.")
        #normalized_data = 0

    # Converts normalized data to binary
    binary_data = (normalized_data > 0.5).astype(int)

    return binary_data

# Load the dataset
data = pd.read_csv('HTRU_2.csv')

# Select the 10000th row as an example and convert it to a NumPy array
data_point = data.iloc[10000, :8].to_numpy(dtype=float)

# Call qml.draw_mpl with the NumPy array
qml.drawer.use_style("sketch_dark")  # Set the plot style

# Drawing the basis_embedding circuit
circuit_amplitude= qml.draw_mpl(amplitude_circuit)(data_point)
circuit_basis= qml.draw_mpl(basis_embedding)(data_point, 8)
plt.show()


#print(data_point_np.shape)  # Should be (8,)
#print(data_point_np)  # Print to see the data point
#print(data_point_np.ndim == 1)  # Should return True

#result, state = amplitude_circuit(f=data_point_np)
#result, state = basis_embedding(data_point_np, 8)

#print(state)
#print("Expectation value:", result)
#print("Quantum state:", state)
