
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import matplotlib.pyplot as plt


print("tensorflow is version: ", tf.__version__)
print("Keras is version: ", keras.__version__)

#Fashion-mnist dataset call
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

#look at this data and see what its dimensions are and what kind it is.
print('x_train_full shape: ', X_train_full.shape, 'x_train_full type: ', X_train_full.dtype)
print('y_train_full shape: ', y_train_full.shape, 'y_train_full type: ', y_train_full.dtype)
print('x_test shape: ', X_test.shape, 'x_test type: ', X_test.dtype)
print('y_test shape:', y_test.shape, 'y_test type', y_test.dtype)

#We select 10,000 train samples and assign them to validation.
#Each pixel in the images has a value between 0 and 255.
# Because we want to use a gradient decsent, we have to bring these values between 0 and 1.
x_valid, x_train = X_train_full[50000:] / 255.0, X_train_full[:50000] / 255.0
y_valid, y_train = y_train_full[50000:], y_train_full[:50000]

#----------Build a network in TensorFlow-------------#
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))#The first layer: the input layer with its own shape size
model.add(keras.layers.Dense(300, activation="relu"))#Second layer: 300 neurons relu activation function
model.add(keras.layers.Dense(100, activation="relu"))#Third layer: 100 neurons of relu activation function
model.add(keras.layers.Dense(10, activation="softmax"))#Last layer: The output layer contains 10 neurons of the softmax activation function
#We check the model layer by layer.
model.summary()
#Define loss function, optimization algorithm and evaluation criteria
model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
#Teach the defined model
history = model.fit(x_train, y_train, epochs=30, validation_data=(x_valid, y_valid))
#Draw relevant diagrams
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()
#Investigate network performance on test data
model.evaluate(X_test/255, y_test)













