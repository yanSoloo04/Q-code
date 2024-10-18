"""
This file contains the run_MLPC and draw the confusion matrix. 
If you want different parameters, just change the call to MLPC classifier on line 31.
"""
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_MLPC(parameters:NDArray, labels:NDArray, matrix:bool = False, filename:str = '')->float:
    """
    Runs the mlpc with the given dataset and returns the accuracy score of the classification using the mlpc method.
    Args:
    parameters (NDArray): the parameters for each given labels in the form of a 2D numpy array
    labels (NDArray): the labels corresponding to the parameters in the form of a 1D numpy array
    matrix (bool): the users specifies wheter he wants to print the confusion matrix or not.
    filename (str): the name of the file for the confusion matrix to be saved as

    Returns:
    the accuracy score after using the MLPC method to classify the datas in the form of a float object.
    """
    #scaling the datas and splitting them into testing datas and training datas.
    scaler = StandardScaler().fit(parameters)
    X_scaled = scaler.transform(parameters)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2)

    #training our classifier
    mlp = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=1000, activation='relu', solver='adam', alpha=0.001)
    mlp.fit(X_train, y_train)

    #predicting the labels with the testing parameters
    print('Classifying the datas...')
    predictions = mlp.predict(X_test)

    #calculating the accuracy score and printing the labels for manual verification
    acc = accuracy_score(y_test, predictions)
    print('Actual labels: ', y_test)
    print('Predicted labels: ', predictions)

    #saving the confusion matrix if the user wants to
    if matrix == True:
        print('Constructing the matrix...')
        cm = confusion_matrix(y_test, predictions)
        print('Drawing the matrix...')
        draw_confusion_matrix(cm, filename)
        print('matrix has been draw')
        
    return acc

def draw_confusion_matrix(matrix: NDArray, file_name:str):
    """
    This functions draws the confusion matrix and saves it as the filename on the same folder as the python script
    Args:
    matrix (NDArray): the array containing the confusion matrix to print
    file_name (str): the name of the file for the confusion matrix to be saved as
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='coolwarm', aspect='equal')
    plt.title( "Confusion Matrix")
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks([0, 1], ["Non-Pulsar", "Pulsar"])
    plt.yticks([0, 1], ["Non-Pulsar", "Pulsar"])
    plt.savefig(file_name)

