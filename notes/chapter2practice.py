# Loading the MNIST dataset in Keras
from keras.datasets import mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# Reshape the input data. Reshape and change type
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

# Building the network architecture
from keras import models
from keras import layers

