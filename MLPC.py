from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data import get_csv_file, get_samples, get_xlsx_file


def run_MLPC(parameters:NDArray, labels:NDArray, matrix:bool )->int:
    """
    Runs the mlpc with the given dataset and returns the accuracy score of the classification using the mlpc method.
    Args:
    parameters (NDArray): the parameters for each given labels in the form of a 2D numpy array
    labels (NDArray): the labels corresponding to the parameters in the form of a 1D numpy array
    matrix (bool): the users specifies wheter he wants to print the confusion matrix or not.

    Returns:
    the accuracy score after using the MLPC method to classify the datas.
    """
    #scaling the datas and splitting them into testing datas and training datas.
    scaler = StandardScaler().fit(parameters)
    X_scaled = scaler.transform(parameters)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.2)

    #training our classifier
    mlp = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=1000, activation='relu', solver='adam', alpha=0.001)
    mlp.fit(X_train, y_train)

    #predicting the labels with the testing parameters
    predictions = mlp.predict(X_test)

    #calculating the accuracy score and printing the labels for manual verification
    acc = accuracy_score(y_test, predictions)
    print('Actual labels: ', y_test)
    print('Predicted labels: ', predictions)

    if matrix == True:
        cm = confusion_matrix(y_test, predictions)
        draw_confusion_matrix(cm, "Confusion Matrix")
    return acc

def draw_confusion_matrix(matrix: NDArray, title: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='coolwarm', aspect='equal')
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks([0, 1], ["Non-Pulsar", "Pulsar"])
    plt.yticks([0, 1], ["Non-Pulsar", "Pulsar"])
    plt.plot()

x = get_xlsx_file('Dry_Bean_Dataset.xlsx')
X, y = get_samples(x, 50, ['SIRA', 'DERMASON'])
accuracy = run_MLPC(X, y, True) 
print(accuracy)
