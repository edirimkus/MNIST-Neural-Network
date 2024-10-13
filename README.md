# MNIST Neural Network

## Overview
This Python script uses TensorFlow to build and train a neural network for classifying handwritten digits from the MNIST dataset. It demonstrates the fundamental steps in loading data, preprocessing, building a model, training, and evaluating performance.

## Script Breakdown
1. **Import Libraries:**
   Imports necessary libraries for data handling, neural network building, and training.
   ```python
   import tensorflow as tf
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense
   from tensorflow.keras.optimizers import Adam
   from tensorflow.keras.losses import MeanSquaredError
   from tensorflow.keras.datasets import mnist
   import numpy as np
   ```

2. **Load Dataset**: Loads the MNIST dataset and normalizes the pixel values.
   ```python
   (x_train, y_train), (x_test, y_test) = mnist.load_data()
   x_train, x_test = x_train / 255.0, x_test / 255.0
   ```

3. **Flatten the Images**: Reshapes the images to a 1D array.
   ```python
   x_train = x_train.reshape((-1, 28*28))
   x_test = x_test.reshape((-1, 28*28))
   ```
   
4. **Model Architecture**: Defines a sequential model with three dense layers.
   ```python
   model = Sequential([
       Dense(128, activation='relu', input_shape=(28*28,)),
       Dense(64, activation='relu'),
       Dense(10, activation='softmax')
   ])
   ```
   
5. **Compile the Model**: Specifies the optimizer, loss function, and metrics.
   ```python
   model.compile(optimizer=Adam(), loss=MeanSquaredError(), metrics=['accuracy'])
   ```
   
6. **Train the Model**: Trains the model on the training data.
   ```python
   model.fit(x_train, tf.keras.utils.to_categorical(y_train), epochs=5, batch_size=32)
   ```
   
7. **Evaluate the Model**: Evaluates the model on the test data and prints the loss and accuracy.
   ```python
   loss, accuracy = model.evaluate(x_test, tf.keras.utils.to_categorical(y_test))
   print(f'Loss: {loss}, Accuracy: {accuracy}')
   ```

## Usage

1. **Run the Script**: Execute the script in a Python environment with the necessary libraries installed.
   ```batch
   python mnist_neural_network.py

   ```

## License
This script is licensed under the MIT License. See the LICENSE file for details.


