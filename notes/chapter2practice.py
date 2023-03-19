# LOADING THE MNIST DATASET IN KERAS
from keras.datasets import mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# Reshape the input data. Reshape and change type
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255



# BUILDING THE NETWORK ARCHITECTURE
from keras import models
from keras import layers

network = models.Sequential()
# The networkconsits of a chain of two Dense layers,each one with itsown weights
# W0, with relu activation function and a input shape of 28*28
network.add(layers.Dense(512, activation='relu',input_shape=(28*28,)))
# W1,with softmax activation function. Output of vector of 10 probability scores
network.add(layers.Dense(10,activation='softmax'))


