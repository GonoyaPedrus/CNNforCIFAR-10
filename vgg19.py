import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix

cifar10 = tf.keras.datasets.cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

y_train, y_test = y_train.flatten(), y_test.flatten()

fig, ax = plt.subplots(5, 5)
k = 0

for i in range(5):
    for j in range(5):
        ax[i][j].imshow(x_train[k], aspect='auto')
        k += 1

plt.show()

K = len(set(y_train))

print("number of classes:", K)

i = Input(shape=x_train[0].shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dropout(0.2)(x)

x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(K, activation='softmax')(x)

model = Model(i, x)

model.summary()

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

r = model.fit(
x_train, y_train, validation_data=(x_test, y_test), epochs=50)

batch_size = 1024 #nbre d'images pour chaque itération
data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

train_generator = data_generator.flow(x_train, y_train, batch_size)
steps_per_epoch = x_train.shape[0] // batch_size

r = model.fit(train_generator, validation_data=(x_test, y_test),
steps_per_epoch=steps_per_epoch, epochs=50)

plt.plot(r.history['accuracy'], label='acc', color='red')
plt.plot(r.history['val_accuracy'], label='val_acc', color='green')
plt.legend()

labels = '''airplane automobile bird cat deer dog frog horse ship truck'''.split()

image_number = 3

plt.imshow(x_test[image_number])

n = np.array(x_test[image_number])

p = n.reshape(1, 32, 32, 3)

predicted_label = labels[model.predict(p).argmax()]

original_label = labels[y_test[image_number]]

print("Original label is {} and predicted label is {}".format(
original_label, predicted_label))

model.save('modeltest.h5')

#tracer graphe de toutes les métriques
# Calcul de la matrice de confusion
cm = confusion_matrix(y_train, y_test)

# Affichage de la matrice de confusion
plt.matshow(cm, cmap='Blues')

# Ajout de la légende
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

# Ajout des annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(cm[i][j]), horizontalalignment='center', verticalalignment='center')

# Ajout des labels d'axes
plt.xlabel('Prédictions')
plt.ylabel('Valeurs réelles')

# Affichage de la figure
plt.show()