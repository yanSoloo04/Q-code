import numpy as np
import pandas as pd
import random
import pennylane as qml
import time
import matplotlib as plt
from pennylane import AngleEmbedding
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

train_size = 10 # Specify the size of the training set.

n_samples = int(train_size/2)

test_size = 10 # Specify the size of the test set.

N_readings = 1 # Number of runs for averaging and error calculation.

for i in range(0,N_readings):

    # Data:

    # We randomise the seed every run and average over 5 separate runs.

    seed = np.random.randint(0,100)

    # Read data from dataset:

    read = pd.read_csv("HTRU_2.csv")

    header = read.columns

    data_with_labels = read.values

    data_with_labels = np.delete(data_with_labels, obj = 0, axis = 0) # Delete the title row.

    # Test train split:

    X = np.delete(data_with_labels, obj = 8, axis = 1)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, np.pi))
    X = min_max_scaler.fit_transform(X)

    Y = np.delete(data_with_labels, obj = [0,1,2,3,4,5,6,7], axis = 1)
    Y = Y[:,0]

    # Test-Train split:

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = test_size, stratify=Y, random_state=seed)

    # Balanced Training set:

    pos_list_features = []
    neg_list_features = []
    pos_list_labels = []
    neg_list_labels = []

    for i in np.arange(0, len(X_train)):
        if (Y_train[i] == 1.0):
            pos_list_features.append(X_train[i])
            pos_list_labels.append(Y_train[i])
        if (Y_train[i] == 0.0):
            neg_list_features.append(X_train[i])
            neg_list_labels.append(Y_train[i])

    pos_list_features = np.array(pos_list_features)
    neg_list_features = np.array(neg_list_features)
    pos_list_labels = np.array(pos_list_labels)
    neg_list_labels = np.array(neg_list_labels)

    # Random Sampling from X_train

    pos_id = random.sample(range(0, pos_list_features.shape[0] - 1), n_samples) # Randomly sample index values to ID pulsars
    neg_id = random.sample(range(0, neg_list_features.shape[0] - 1), n_samples) # and none pulsars.

    rows_pos_features = pos_list_features[pos_id, :] # The ID of the features and labels should be the same for each sample.
    rows_pos_labels = pos_list_labels[pos_id]
    rows_neg_features = neg_list_features[neg_id, :]
    rows_neg_labels = neg_list_labels[neg_id]

    sample_data_features = np.concatenate((rows_pos_features, rows_neg_features)) # These are the new X_train and Y_train.
    sample_data_labels = np.concatenate((rows_pos_labels, rows_neg_labels))

    temp_length = len(sample_data_features)

    permutation_index = np.random.permutation(temp_length) # Generate a random permutation index

    X_train_balanced = sample_data_features[permutation_index] # Apply the permutation index to both arrays
    Y_train_balanced = sample_data_labels[permutation_index]
    
################################################################################################################
################################################################################################################
################################################################################################################

    ### QSVM ### 

    # Circuit and Kernel Matrix:

    n_qubits = len(X_train[0]) # Qubits = feature amount.

    dev_kernel = qml.device("default.qubit", wires=n_qubits)

    projector = np.zeros((2**n_qubits, 2**n_qubits)) # Zero matrix for initial state measurement.
    projector[0, 0] = 1

    @qml.qnode(dev_kernel)
    def kernel(x1, x2):
        """The quantum kernel."""
        AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

    def kernel_matrix(A, B):
        """Compute the matrix whose entries are the kernel
           evaluated on pairwise data from sets A and B."""
        return np.array([[kernel(a, b) for b in B] for a in A])

    # Specify drawer:

    qml.drawer.use_style("black_white")

    # Draw the circuit:

    fig, ax = qml.draw_mpl(kernel)(X_train_balanced, X_train_balanced) # The input here is incorrect - this is merely for visualisation.

    #########################################################################################

    # Training:

    st = time.time()
    
    svm = SVC(kernel=kernel_matrix).fit(X_train_balanced, Y_train_balanced)

    et = time.time()
    
    pred_st = time.time()

    predictions = svm.predict(X_test)
    
    pred_et = time.time()

    evaluations = dev_kernel.num_executions

    print('Amount of Quantum Evaluations:', evaluations)

    cf_matrix = confusion_matrix(Y_test, predictions)

    ###########################################################################################

    # Plot the CM:

    #cf_matrix = cf_matrix/np.sum(cf_matrix) # 

    fig, ax = plt.subplots()
    im = ax.imshow(cf_matrix, cmap = 'Greens')

    # Add the values to each pixel:
    for i in range(cf_matrix.shape[0]):
        for j in range(cf_matrix.shape[1]):
            text = ax.text(j, i, cf_matrix[i, j], ha="center", va="center", color="r", fontsize = 14)

    # Remove the x and y labels and numbers
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    ax.set_xticklabels(["Non-pulsar", "Pulsar"], fontsize = 14)
    ax.set_yticklabels(["Non-pulsar", "Pulsar"], fontsize = 14)
    ax.set_xlabel("")
    ax.set_ylabel("")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Amount of Classifications", rotation=270, labelpad=15, fontsize = 14)

    ###########################################################################################

    # Metrics:

    true_pos = cf_matrix[1][1]
    true_neg = cf_matrix[0][0]
    false_pos = cf_matrix[0][1]
    false_neg = cf_matrix[1][0]

    print("True Positives", true_pos)
    print("True Negatives", true_neg)
    print("False Positives", false_pos)
    print("False Negatives", false_neg)
    
    training_time = et - st
    print("Training Time", training_time, "seconds.")
    
    prediction_time = pred_et - pred_st
    print("Prediction Time", prediction_time, "seconds")