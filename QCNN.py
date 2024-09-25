import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt


def pooling_circuit(nb_qubits: int)->int:
    assert(nb_qubits%2 == 0)
    m = int(nb_qubits/2)
    for i in range(m):
        j = nb_qubits-1-i
        qml.CNOT(wires = [j, i])
    return m

def convution_circuit(nb_qubits: int):
    qml.RY(1.2, wires = nb_qubits-1)





test = pooling_circuit(8)
# fig, ax = qml.draw_mpl(pooling_circuit)(4)
# fig.show()

y= 1+2
