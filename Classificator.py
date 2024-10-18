from QCNN import run_QCNN
from VQC import run_VQC
from QSVM import run_QSVM
from MLPC import run_MLPC
from data import get_csv_file, get_xlsx_file, get_samples
import numpy as np


HTRU_2_dataset = get_csv_file('HTRU_2.csv')
Dry_Bean_Dataset = get_xlsx_file('Dry_Bean_Dataset.xlsx')

nb_data = 60
beans_labels = ['SIRA', 'DERMASON']

HTRU_2_parameters, HTRU_2_labels = get_samples(HTRU_2_dataset, nb_data, [0, 1])
Dry_Bean_parameters, Dry_Bean_labels = get_samples(Dry_Bean_Dataset, nb_data, beans_labels)

embedding_method = 'amplitude'
rotation = 'Y'

batch_size = 30
ansatz = 'layer'


results = np.zeros((2, 4))
####TESTING THE HTRU_2 DATASET##################
print('#####################################-------------testing the HTRU_2 dataset--------------##########################################')

print('Running MLPC...')
mlpc_accuracy = run_MLPC(HTRU_2_parameters, HTRU_2_labels)
print('MLPC ACCURACY:', mlpc_accuracy)
results[0, 0] = mlpc_accuracy

print('Running the QSVM...')
qsvm_accuracy = run_QSVM(HTRU_2_parameters, HTRU_2_labels, embedding_method, rotation)
print('QSVM ACCURACY:', qsvm_accuracy)
results[0, 1] = qsvm_accuracy

print('Running the VQC...')
vqc_accuracy = run_VQC(HTRU_2_dataset, nb_data, batch_size, [0, 1], ansatz, embedding_method, rotation)
print('VQC ACCURACY:', vqc_accuracy)
results[0, 2] = vqc_accuracy

print('Running the QCNN...')
qcnn_accuracy = run_QCNN(HTRU_2_dataset, nb_data, batch_size, [0, 1], embedding_method, rotation)
print('QCNN ACCURACY:', qcnn_accuracy)
results[0, 3] = qcnn_accuracy




####TESTING THE DRY BEAN DATASET#################
print('#####################################-------------testing the Dry_Bean_Dataset--------------##########################################')

print('Running MLPC...')
mlpc_accuracy = run_MLPC(Dry_Bean_parameters, Dry_Bean_labels)
print('MLPC ACCURACY:', mlpc_accuracy)
results[1, 0] = mlpc_accuracy

print('Running the QSVM...')
qsvm_accuracy = run_QSVM(Dry_Bean_parameters, Dry_Bean_labels, embedding_method, rotation)
print('QSVM ACCURACY:', qsvm_accuracy)
results[1, 1] = qsvm_accuracy

print('Running the VQC...')
vqc_accuracy = run_VQC(Dry_Bean_Dataset, nb_data, batch_size, beans_labels, ansatz, embedding_method, rotation)
print('VQC ACCURACY:', vqc_accuracy)
results[1, 2] = vqc_accuracy

print('Running the QCNN...')
qcnn_accuracy = run_QCNN(Dry_Bean_Dataset, nb_data, batch_size, beans_labels, embedding_method, rotation)
print('QCNN ACCURACY:', qcnn_accuracy)
results[1, 3] = qcnn_accuracy


print('##################################################----------END OF THE EXECUTION-----------########################################')

print('accuracy array: ', results)