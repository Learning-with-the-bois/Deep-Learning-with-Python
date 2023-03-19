# ==== LOADING THE MNIST DATASET IN KERAS ==== #
from keras.datasets import mnist

(train_images, train_labels),(test_images, test_labels) = mnist.load_data()

# ==== PREPARING THE INPUT DATA ====#
# Reshape the input data. Reshape and change type
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255



# ==== BUILDING THE NETWORK ARCHITECTURE====#
from keras import models
from keras import layers

network = models.Sequential()
# The network consists of a chain of two Dense layers,each one with itsown weights
# W0, with relu activation function and a input shape of 28*28, returns a 512-dimentional vector
network.add(layers.Dense(512, 
                         activation='relu',input_shape=(28*28,)))
# W1,with softmax activation function. Output of vector of 10 probability scores
network.add(layers.Dense(10,
                         activation='softmax'))


# ==== COMPILATION ==== #
# "categorycal_crossentropy" is the loss function that's used as a feedback signal for learning the weight tensors, andwhich the training phase will attempt to minimize.
# "rmsprop" is the optimizer with the exact rules governing a specific use of gradient descent.
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# =================================== #
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# =================================== #

# ==== TRAINING LOOP ==== #
# The network iterates in mini-batches of 128 samples, 5 times over
# Each iteration over all the training data is called an epoch
network.fit(train_images,train_labels, epochs=5,batch_size=128)
