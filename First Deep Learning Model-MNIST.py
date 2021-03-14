



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from  tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import sys


physical_devices = tf.config.list_logical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)




(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)

x_train = x_train.reshape(-1, 28*28).astype("float32")/255.0
x_test = x_test.reshape(-1, 28*28).astype("float32")/255.0


#Sequential API(map is one input to one output, easy but not flexible)

#First Method of Sequential Way
model = keras.Sequential(
    [
        keras.Input(28*28),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(10),
    ]

)


# print(model.summary())
# import sys
# sys.exit()


#Second Method of Sequential Way
model = keras.Sequential()
model.add(keras.Input(shape=(28*28)))
model.add(layers.Dense(512, activation='relu', name='first_layer'))
model.add(layers.Dense(256, activation='relu', name='second_layer'))
model.add(layers.Dense(10))
model.summary()


model_features = keras.Model(inputs=model.inputs, outputs=[layer.output for layer in model.layers])
fetures = model_features.predict(x_train)
for feature in fetures:
    print(feature.shape)


#Functional API(you can handle multiple input and multiple output. so is flexible)
inputs = keras.Input(shape=(28*28))
x = layers.Dense(512, activation='relu')(inputs)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)



model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(lr=0.001),
    metrics=['accuracy'],
)

model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)





