# Qcode: Kernel Method

Cette branche contient le code pour l'implémentation de la méthode à noyaux, une technique clé dans le machine learning quantique. Nous avons exploré plusieurs approches, notamment :

- **QSVM (Quantum Support Vector Machine)** : Une méthode utilisant des vecteurs de support quantiques pour la classification.
- **QCNN (Quantum Convolutional Neural Network)** : Un réseau de neurones convolutionnels quantiques, adapté à l'analyse de données structurées.
- **MLPC (Multi-Layer Perceptron Classifier)** : Un classificateur basé sur un perceptron multicouche, utilisé pour des problèmes de classification complexes.
**Embedding** : Techniques d'encodage des données pour les mapper dans un espace quantique. Nous avons implémenté plusieurs types d'embedding :
  - **Angle Embedding** : Utilisation d'angles pour représenter les données.
  - **Amplitude Embedding** : Encodage des amplitudes des états quantiques.
  - **Basis Embedding** : Mapping des données dans la base quantique.

## Objectif
L'objectif de cette branche est d'utiliser différentes méthodes à noyaux pour améliorer la classification des données et mieux séparer les différentes classes.

## Contenu
- **`QSVM.py`** : Implémentation spécifique pour QSVM.
- **`QCNN.py`** : Script pour QCNN.
- **`MLPC.py`** : Script pour le classificateur MLPC.
- **`embedding.py`** : Fonctions pour l'embedding des données, y compris l'angle embedding, l'amplitude embedding et le basis embedding.

