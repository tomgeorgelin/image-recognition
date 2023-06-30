# Projet de Reconnaissance d'Images avec Réseau Neuronal

Ce projet est une démonstration de reconnaissance d'images en utilisant le dataset MNIST et un réseau neuronal réécrit à la main. L'objectif est de prédire correctement les chiffres écrits à la main à partir des images fournies dans le dataset.

## Données

Les données utilisées pour ce projet sont extraites du célèbre dataset MNIST, qui est une base de données de chiffres manuscrits. Les images sont en niveaux de gris et ont une taille de 28x28 pixels. Chaque image représente un chiffre de 0 à 9.

Les données sont stockées dans le dossier `data/`. Voici les fichiers qu'il vous faudra télécharger [(lien disponible ici)](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) et placer dans le dossier :

-   `mnist_train.csv` : fichier contenant les images d'enrtainement.
-   `mnist_test.csv` : fichier contenant les images de test.

## Exécution du Projet

Pour exécuter le projet, assurez-vous d'avoir Python installé sur votre système. Suivez les étapes ci-dessous :

1. Clonez ce dépôt sur votre machine :

```
http :
git clone https://github.com/tomgeorgelin/image-recognition.git
ssh :
git clone git@github.com:tomgeorgelin/image-recognition.git
```

2. Accédez au répertoire du projet :
   `cd image-recognition`

3. Installez les dépendances requises :
   `pip3 install numpy pandas matplotlib`

4. Lancez le script principal :
   `python main.py`

Le script chargera les données, entraînera le réseau neuronal et effectuera des prédictions sur les images de test en fonction de l'index fourni par l'utilisateur. Les résultats seront affichés dans la console et l'image à l'écran.

Lors des tests, pour 1000 itérations le réseau était précis à ~85%.

## Auteur

Ce projet a été développé par [Tom Georgelin](https://github.com/tomgeorgelin) et est basé sur une vidéo de [Samson Zhang](https://www.youtube.com/watch?v=w8yWXqWQYmU&ab_channel=SamsonZhang).

N'hésitez pas à me contacter si vous avez des questions ou des commentaires !
