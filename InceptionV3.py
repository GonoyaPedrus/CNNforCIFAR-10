import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import time

# Chargement de la base de données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalisation des images
x_train = x_train / 255.0
x_test = x_test / 255.0

# Encodage one-hot des étiquettes
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Chargement du modèle InceptionV1 avec des images de taille 32x32
input_layer = Input(shape=(32, 32, 3))
x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Conv2D(64, (1, 1), strides=(1, 1), padding='valid', activation='relu')(x)
x = Conv2D(192, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Conv2D(192, (1, 1), strides=(1, 1), padding='valid', activation='relu')(x)
x = Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = Conv2D(256, (1, 1), strides=(1, 1), padding='valid', activation='relu')(x)
x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
x = Flatten()(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(4096, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(10, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)

# Entraînement du modèle
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
start_time = time.time()
model.fit(x_train, y_train, epochs=25, batch_size=64, validation_data=(x_test, y_test))
end_time = time.time()

# Prédiction sur l'ensemble de test
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Calcul des mesures de performance
acc = np.sum(y_pred_classes == y_true) / np.size(y_true)
err = 1 - acc
cm = confusion_matrix(y_true, y_pred_classes)
cr = classification_report(y_true, y_pred_classes, target_names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])
fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1], pos_label=1)


# Affichage des résultats
print("Temps d'entraînement: {:.2f}s".format(end_time - start_time))
print("Précision: {:.2f}%".format(acc * 100))
print("Erreur: {:.2f}%".format(err * 100))
print("Matrice de confusion:\n", cm)
print("Rapport de classification:\n", cr)
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()



model.save('inception_model.h5')


