import numpy as np
import pennylane as qml
from pennylane import AngleEmbedding
from numpy.typing import NDArray
from typing import Callable
import matplotlib.pyplot as plt



device = qml.device("default.qubit")
@qml.qnode(device)
def kernel(x: NDArray, y: NDArray, embedding: Callable):
    assert(len(x) == len(y))
    nb_qubits = len(x)
    projector = np.zeros((2**nb_qubits, 2**nb_qubits))
    projector[0, 0]=1

    embedding(features = x, wires = range(nb_qubits))
    qml.adjoint(embedding(features = y, wires = range(nb_qubits)))
    return qml.expval(qml.Hermitian(projector, wires = range(nb_qubits)))



data = np.genfromtxt('HTRU_2.csv', delimiter=',', skip_header=0)  
x_train = data[:, :-1]

nb_lignes = data.shape[0]
solution = data[:, -1]



for i in range(len(x_train)):
    m = np.max(x_train[i])
    x_train[i] = x_train[i]/m*np.pi/2  



nb_qubits = len(x_train[0])
projector = np.zeros((2**nb_qubits, 2**nb_qubits))
projector[0, 0]=1


essai = x_train[1:50, :]

matrix = np.array([[kernel(a, b, AngleEmbedding) for b in essai] for a in essai])

fig, ax = plt.subplots()
im = ax.imshow(matrix, cmap = 'YlOrBr')




ax.set_xlabel("x")
ax.set_ylabel("y")

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Amount of Classifications", rotation=270, labelpad=15, fontsize = 14)

plt.plot()

y = 1+2