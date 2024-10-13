import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.datasets import mnist
import numpy as np

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten the images
x_train = x_train.reshape((-1, 28*28))
x_test = x_test.reshape((-1, 28*28))

# Model architecture
model = Sequential([
    Dense(128, activation='relu', input_shape=(28*28,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])

# Train the model
model.fit(x_train, tf.keras.utils.to_categorical(y_train), epochs=5, batch_size=32)

# Evaluate the model
loss, accuracy = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test))

print(f'Loss: {loss}, Accuracy: {accuracy}')
