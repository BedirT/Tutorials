import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, x_train.shape[1]*x_train.shape[2]).astype('float32')/255.0
x_test = x_test.reshape(-1, x_test.shape[1]*x_test.shape[2]).astype('float32')/255.0

# Sequential API
# (Very Convenient, not flexible - 1 input to 1 output mapping)
model = keras.Sequential(
    [
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ]
)

# 2nd version
# model = keras.Sequential()
# model.add(keras.Input(shape=(784)))
# model.add(layers.Dense(512, activation='relu'))
# print(model.summary())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(10))

########################################
# Functional API (A bit more flexible) #
########################################
# inputs = keras.Input(shape=(28*28))
# x = layers.Dense(512, activation='relu', name='First_layer')(inputs)
# x = layers.Dense(256, activation='relu')(x)
# outputs = layers.Dense(10, activation='softmax')(x)
# model = keras.Model(inputs=inputs, outputs=outputs)
# print(model.summary())

########################################

# Getting the outputs of the middle layers
# model = keras.Model(inputs=model.inputs,
#                     outputs=model.layers[-3].output)
# model = keras.Model(inputs=model.inputs,
#                     outputs=model.get_layer('First_layer').output)
# model = keras.Model(inputs=model.inputs,
#                     outputs= [layer.output for layer in model.layers])

# features = model.predict(x_train)
# for feature in features:
#     print(feature.shape)

model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    # from_logits -> First softmax since we did not add an activation
    optimizer = keras.optimizers.Adam(lr=0.001),
    # optimizer = keras.optimizers.Adagrad(lr=0.001),
    metrics = ['accuracy']
)

model.fit(x_train, y_train, batch_size=32, epochs=20)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)