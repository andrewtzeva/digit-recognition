import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# 28x28 images of hand-written digits 0-9
mnist = tf.keras.datasets.mnist

# Loading the dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Data normalization
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Building the model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)


# Predictions
predictions = model.predict([x_test])

print(np.argmax(predictions[0]))

plt.imshow(x_test[0])
plt.show()







