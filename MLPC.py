import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from data import get_csv_file, get_samples, get_xlsx_file

x = get_xlsx_file('Dry_Bean_Dataset.xlsx')
X, y = get_samples(x, 50, ['SIRA', 'DERMASON'])

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)


mlp = MLPClassifier(hidden_layer_sizes=(200, 200), max_iter=1000, activation='relu', solver='adam', alpha=0.001)
mlp.fit(X_train, y_train)


predictions = mlp.predict(X_test)


print(predictions, y_test)
print("Accuracy:", accuracy_score(y_test, predictions))


def draw_confusion_matrix(matrix: NDArray, title: str):
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, interpolation='nearest', cmap='coolwarm', aspect='equal')
    plt.title(title)
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.xticks([0, 1], ["Non-Pulsar", "Pulsar"])
    plt.yticks([0, 1], ["Non-Pulsar", "Pulsar"])
    plt.show()

# cm = confusion_matrix(y_test, predictions)
# draw_confusion_matrix(cm, "Confusion Matrix")
