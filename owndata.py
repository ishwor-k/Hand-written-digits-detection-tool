from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image
from keras import backend as K
import numpy as np
import os

seed = 7
np.random.seed(seed)

#load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Reshaping to format which CNN expects (batch, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1).astype('float32')

#To load images to features and labels
def load_images_to_data(image_label, image_directory, features_data, label_data):
    list_of_files = os.listdir(image_directory)
    for file in list_of_files:
        image_file_name = os.path.join(image_directory, file)
        if ".png" in image_file_name:
            img = Image.open(image_file_name).convert("L")
            img = np.resize(img, (28,28,1))
            im2arr = np.array(img)
            im2arr = im2arr.reshape(1,28,28,1)
            features_data = np.append(features_data, im2arr, axis=0)
            label_data = np.append(label_data, [image_label], axis=0)
    return features_data, label_data


# Load your own images to training and test data. Loading images of 0.
X_train, y_train = load_images_to_data('0', 'data/mnist_data/train1/0', X_train, y_train)
X_test, y_test = load_images_to_data('0', 'data/mnist_data/validation1/0', X_test, y_test)

# Load your own images to training and test data. Loading images of 1.
X_train, y_train = load_images_to_data('1', 'data/mnist_data/train1/1', X_train, y_train)
X_test, y_test = load_images_to_data('1', 'data/mnist_data/validation1/1', X_test, y_test)

# Load your own images to training and test data. Loading images of 2.
X_train, y_train = load_images_to_data('2', 'data/mnist_data/train1/2', X_train, y_train)
X_test, y_test = load_images_to_data('2', 'data/mnist_data/validation1/2', X_test, y_test)

# Load your own images to training and test data. Loading images of 3.
X_train, y_train = load_images_to_data('3', 'data/mnist_data/train1/3', X_train, y_train)
X_test, y_test = load_images_to_data('3', 'data/mnist_data/validation1/3', X_test, y_test)

# Load your own images to training and test data. Loading images of 4.
X_train, y_train = load_images_to_data('4', 'data/mnist_data/train1/4', X_train, y_train)
X_test, y_test = load_images_to_data('4', 'data/mnist_data/validation1/4', X_test, y_test)

# Load your own images to training and test data. Loading images of 5.
X_train, y_train = load_images_to_data('5', 'data/mnist_data/train1/5', X_train, y_train)
X_test, y_test = load_images_to_data('5', 'data/mnist_data/validation1/5', X_test, y_test)

# Load your own images to training and test data. Loading images of 6.
X_train, y_train = load_images_to_data('6', 'data/mnist_data/train1/6', X_train, y_train)
X_test, y_test = load_images_to_data('6', 'data/mnist_data/validation1/6', X_test, y_test)

# Load your own images to training and test data. Loading images of 7.
X_train, y_train = load_images_to_data('7', 'data/mnist_data/train1/7', X_train, y_train)
X_test, y_test = load_images_to_data('7', 'data/mnist_data/validation1/7', X_test, y_test)

# Load your own images to training and test data. Loading images of 8.
X_train, y_train = load_images_to_data('8', 'data/mnist_data/train1/8', X_train, y_train)
X_test, y_test = load_images_to_data('8', 'data/mnist_data/validation1/8', X_test, y_test)

# Load your own images to training and test data. Loading images of 9.
X_train, y_train = load_images_to_data('9', 'data/mnist_data/train1/9', X_train, y_train)
X_test, y_test = load_images_to_data('9', 'data/mnist_data/validation1/9', X_test, y_test)

# normalize inputs from 0-255 to 0-1
X_train/=255
X_test/=255
input_shape = (28, 28, 1)

# one hot encode
number_of_classes = 10
y_train = np_utils.to_categorical(y_train, number_of_classes)
y_test = np_utils.to_categorical(y_test, number_of_classes)

# create model of our neural network.
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(number_of_classes, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=24, batch_size=256)

# Save the model
model.save('CNN2.h5')

metrics = model.evaluate(X_test, y_test, verbose=0)
print("Metrics(Test loss & Test Accuracy): ")
print(metrics)
print("Saving the model as CNN2.h5")
