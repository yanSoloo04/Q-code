
# Qcode: Quantum Machine Learning

Bienvenue dans le dépôt GitHub de **Qcode** ! Nous sommes une équipe passionnée par l'apprentissage quantique et explorons différentes méthodes de machine learning quantique à travers ce projet. L'objectif est de comparer les performances de techniques telles que les **méthodes à noyaux** et les **classificateurs variationnels quantiques (VQC)** sur des ensembles de données.

## Structure du Projet
Le dépôt est divisé en plusieurs branches, chacune correspondant à une partie spécifique de notre travail :

- **`data`** : Contient les ensembles de données utilisés pour l'entraînement et la validation.
- **`documentation`** : Documentation détaillée du projet, y compris le rapport technique.
- **`kernel_method`** : Implémentation de l'approche par méthode à noyaux pour la classification quantique.
- **`vqc_method`** : Implémentation de classificateurs variationnels quantiques.
- **`test`** : Scripts et fichiers de tests pour vérifier l'intégrité des différentes méthodes.

## Utilisation
Pour tester les différentes méthodes proposées dans ce projet, une fonction run est définie dans chaque méthodes. Le fichier classificator.py est un bon début pour tester les méthodes avec différents paramètres. Pour comprendre le fonctionnement général de l'implémentation, un tutoriel jupyter notebook est diponible. 

##Fichiers contenus dans le main dans le fichier 'code'
- **'QSVM.py'** : implémentation de la méthode de noyaux quantique
- **'QCNN.py'** : implémentation de la méthode du réseau neuronnal convolutif quantique
- **'VQC.py'** : implémentation de la méthode de classificateur variationnel quantique
- **'MLPC.py'** : implémentation d'une méthode d'apprentissage machine classique
- **'Classificator.py'** : fichier test qui teste toutes les méthodes en fonction des datasets fournis et des différents paramètres fournis par l'utilisateur.
- **'requirement.txt'** : fichier texte avec les commandes d'installation des différents modules requis pour rouler le code

Des exemples graphiques se trouvent dans le classeur dessins_circuit.




