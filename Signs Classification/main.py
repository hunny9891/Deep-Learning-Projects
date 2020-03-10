#%%
import os
import pandas as pd
import tensorflow as tf

ROOT = os.getcwd()
data_dir = os.path.join(ROOT, 'data')
print(data_dir)

#%%
train_raw = pd.read_csv(os.path.join(data_dir, 'sign_mnist_train.csv'))
test_raw = pd.read_csv(os.path.join(data_dir, 'sign_mnist_test.csv'))

print(train_raw.head())
print(test_raw.head())


#%%
Y_train = train_raw['label']
Y_test = test_raw['label']

X_train = train_raw.drop('label', axis=1)
X_test = test_raw.drop('label',axis=1)

#%%
import numpy as np
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)

print(X_train.shape)
print(X_test.shape)

#%%

X_train = np.reshape(X_train, (27455,28,28))
X_test = np.reshape(X_test, (7172,28,28))

print(X_train.shape)
print(X_test.shape)

#%%
# Expand dimensions
X_train = np.expand_dims(X_train, axis=3)
X_test = np.expand_dims(X_test, axis=3)

print(X_train.shape)
print(X_test.shape)

#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)


#%%
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),

    # tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),

    # tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
    # tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')
])

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])

history = model.fit_generator(train_datagen.flow(X_train,Y_train, batch_size=32),
steps_per_epoch=len(X_train)/ 32,
epochs=50,
validation_data = validation_datagen.flow(X_test, Y_test, batch_size=32),
validation_steps=len(X_test)/ 32)

model.evaluate(X_test, Y_test)

#%%
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

#%%
