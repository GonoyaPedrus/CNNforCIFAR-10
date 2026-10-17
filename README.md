# CNNforCIFAR-10

## Présentation du projet

Projet d'école avec @tombrousd et @loriscristo visant à créer et entraîner nos propres modèles de CNN sur le dataset CIFAR10 et avec une webapp à la clé visant à prédire le contenu des photos de l'user.
Je n'ai pas encore retrouvé l'application web du projet.

## Scripts

| Script | Contenu |
| --- | --- |
| `vgg_like.py` | Petit CNN inspiré de VGG (6 convolutions + BatchNorm), entraîné sans puis avec augmentation de données. Sauvegarde `vgg_like.h5`. |
| `inception_like.py` | Architecture inspirée d'InceptionV1 (tronc GoogLeNet, sans modules Inception) adaptée au 32x32, avec rapport de classification et courbes ROC par classe. Sauvegarde `inception_like.h5`. |
| `resnet50_transfer.py` | Transfer learning à partir de ResNet50 pré-entraîné sur ImageNet. |
| `confusion_matrix.py` | Matrice de confusion sur le jeu de test à partir de `vgg_like.h5`. |

## Utilisation

```bash
pip install -r requirements.txt
python vgg_like.py
python confusion_matrix.py
```

Le dataset CIFAR-10 est téléchargé automatiquement par Keras au premier lancement.
