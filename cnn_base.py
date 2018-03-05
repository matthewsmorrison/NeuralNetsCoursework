
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
from utils.data_utils import get_CIFAR10_data, get_FER2013_data
import cPickle
import numpy
import csv
import scipy.misc
import scipy
from scipy import ndimage
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import dataprocessing




def cnn():

    ep = 0.000001
    p = 0.1
    learn_rate = 0.01
    width, height = 48,48
    cnn = Sequential()


    cnn.add(Convolution2D(64,3,3, padding = 'same',strides = (1,1),border_mode = 'valid', input_shape = (width, height,1)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    cnn.add(Convolution2D(64,3,3, padding = 'same',strides = (1,1)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    cnn.add(Convolution2D(64,3,3, padding = 'same',strides = (1,1)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    cnn.add(Convolution2D(64,3,3, padding = 'same',strides = (1,1)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

    cnn.add(Flatten())
    cnn.add(Dense(256))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(p))
    cnn.add(Dense(512))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(p))
    cnn.add(Dense(7))

    cnn.add(Activation('softmax'))




    A = Adadelta(lr=learn_rate, rho=0.95, epsilon=ep)
    cnn.compile(loss='categorical_crossentropy',optimizer=A,metrics=['accuracy'])

    cnn.summary()

# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

depth = 1
epochs = 20
classes = 7
width = height = 48
batch_size = 128



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH THE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

data = get_FER2013_data()

x_train = data["X_train"].reshape(-1,width, depth)
y_train = np_utils.to_categorical(data["y_train"], classes)
x_val = data["X_val"].reshape(-1,width, depth)
y_train = np_utils.to_categorical(data["y_val"], classes)
x_test = data["X_test"].reshape(-1,width, depth)
y_train = np_utils.to_categorical(data["y_test"], classes)


generate = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    rotation_range=5,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.05,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.05,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

our_model = cnn()
files='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
ckpt = keras.callbacks.ModelCheckpoint(files, monitor = 'val_loss',verbose=1, save_best_only=True, mode='auto')

generate.fit(x_train)
our_model.fit_generator(generate.flow(x_train, y_train), samples_per_epoch = x_train.shape[0], batch_size = batch_size, nb_epoch=epochs,
                        validation_data = (x_val, y_val), callbacks = [ckpt])
