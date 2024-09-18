import numpy as np
import pennylane as qml
from numpy.typing import NDArray


def embedding_circuit(params : NDArray):
    for i, parameter in enumerate(params):
        qml.RY(phi = parameter, wires = i)
    


X=np.array([1, 2, 3, 4, 5, 6, 7, 8])
Y=np.array([1, 3, 2, 4, 5, 6, 7, 8])

dev = qml.device("default.qubit", wires=8)
@qml.qnode(dev)
def QSVM_circuit(X, Y):
    embedding_circuit(X)
    qml.adjoint(embedding_circuit, (Y))
    return qml.expval(qml.PauliZ(i) for i in range(8))

test = QSVM_circuit(X, Y)