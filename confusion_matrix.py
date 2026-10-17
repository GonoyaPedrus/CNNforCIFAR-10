import tensorflow as tf
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

classes = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

# Charger le modèle entraîné par vgg_like.py
model = tf.keras.models.load_model('vgg_like.h5')

# Load in the data
cifar10 = tf.keras.datasets.cifar10

# Distribute it to train and test set
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# Reduce pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# flatten the label values
y_train, y_test = y_train.flatten(), y_test.flatten()

# Prédire les étiquettes pour les données de test
y_pred = model.predict(x_test)

# Convertir les probabilités en étiquettes de classe
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = y_test

# Calculer la matrice de confusion
confusion_mtx = confusion_matrix(y_test_classes, y_pred_classes)

# Afficher la matrice de confusion sous forme graphique
plt.imshow(confusion_mtx, interpolation='nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Prédit')
plt.ylabel('Réel')
plt.show()
