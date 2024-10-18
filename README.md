
# Qcode: Quantum Machine Learning

Bienvenue dans le dépôt GitHub de **Qcode** ! Nous sommes une équipe passionnée par l'apprentissage quantique et explorons différentes méthodes de machine learning quantique à travers ce projet. L'objectif est de comparer les performances de techniques telles que le **Réseau Neuronal Convolutif Quantique**, **Machine à vecteur de support quantique​** , et les **classificateurs variationnels quantiques (VQC)** sur des ensembles de données.

## Structure du Projet
Le dépôt est divisé en plusieurs branches, chacune correspondant à une partie spécifique de notre travail. Mais vous pouvez trouver tous les documents nécessaires sur la branche 'main' :

- **`Code`** : Dossier contenant le code pour les differents approches qml et classique utilisés dans le projet.
- **`data_set`** : Dossier contenant les ensembles de données utilisés pour l'entraînement et la validation.
- **`dessins_circuits`** : Dossier contenant les dessins des circuits obtenus.

## Utilisation
Pour bien comprendre l'implementation des différentes méthodes proposées dans ce projet, vous pouvez commencer par consulter le jupyter_notebook **'tutorial.ipynb'** dans le dossier code. Vous pouvez ensuite importer les librairies utiles qui existent dans **'requirement.txt'**. Puis, consulter **'Classificator.py'** qui teste toutes les méthodes en fonction des datasets fournis et des différents paramètres fournis par l'utilisateur.

##Fichiers contenus dans le dossier 'code'
- **'QSVM.py'** : implémentation de la méthode de machine à vecteur de support quantique.
- **'QCNN.py'** : implémentation de la méthode du réseau neuronnal convolutif quantique.
- **'VQC.py'** : implémentation de la méthode de classificateur quantique variationnel.
- **'MLPC.py'** : implémentation d'une méthode d'apprentissage machine classique ( classificateur perceptron multicouche ).
- **'Classificator.py'** : fichier test qui teste toutes les méthodes en fonction des datasets fournis et des différents paramètres fournis par l'utilisateur.
- **'requirement.txt'** : fichier texte avec les commandes d'installation des différents modules requis pour rouler le code.
- **'data.py'** : fichier qui sert à aller chercher les données d'un fichier et à traiter ces données pour faire des tests.
- **'tutorial.ipynb'** : Cahier Jupyter pour reproduire et comprendre nos implémentations pas à pas.
- **'Dry_Bean_Dataset.xlsx'** : Ensemble de données sur les haricots secs, contenant des caractéristiques pour la classification des différentes variétés de haricots.
- **'HTRU_2.csv'** : Ensemble de données sur les candidats pulsars collectés lors du sondage HTRU. Les pulsars sont un type d'étoile d'un intérêt scientifique considérable. Les candidats doivent être classés en classes pulsars et non pulsars pour faciliter la découverte.

  
Des exemples graphiques se trouvent dans le classeur dessins_circuit.




