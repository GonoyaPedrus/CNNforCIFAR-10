import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.utils import to_categorical

# Charger l'ensemble de données CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Prétraiter les données pour l'adapter au modèle ResNet50
x_train = preprocess_input(x_train)
x_test = preprocess_input(x_test)
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Charger le modèle ResNet50 pré-entraîné sur ImageNet
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32,32,3))

# Ajouter des couches pour adapter le modèle à CIFAR-10
model = tf.keras.models.Sequential()
model.add(base_model)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiler le modèle avec une fonction de perte et un optimiseur
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# Évaluer le modèle sur l'ensemble de test
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print("Indice de précision : %.2f%%" % (accuracy * 100))
