import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10

# Images
# -----------
# 50k training
# 3 channels
# 32 x 32

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

model = keras.Sequential(
    [
        keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding='same', kernel_regularizer=regularizers.l2(0.01), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(2),
        layers.Conv2D(64, 3, padding='same', kernel_regularizer=regularizers.l2(0.01), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(),
        layers.Conv2D(128, 3, padding='same', kernel_regularizer=regularizers.l2(0.01), activation='relu'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(64, kernel_regularizer=regularizers.l2(0.01), activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10)
    ]
)

print(model.summary())

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(x_train, y_train, batch_size=64, epochs=150, verbose=2)
model.evaluate(x_test, y_test, batch_size=64, verbose = 2)