from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import scipy.misc
import platform
import PIL as pillow
from PIL import Image
# import tensorflow as tf

# import tensorflow as tf


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return  pickle.load(f)
    elif version[0] == '3':
        return  pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000,
                     subtract_mean=True):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_FER2013_data(num_training = 28709,num_test = 3589,num_val = 4000):

    if(num_val > num_training*0.2):
        print("num_val size too large, please insert something smaller")

    directory = 'datasets/public'
    train_dir = directory + '/Train/'
    test_dir = directory + '/Test/'

    labels = np.loadtxt(directory + '/labels_public.txt',skiprows=1,delimiter=',',usecols=1,dtype='int')

#    data = dict.fromkeys(['X_train','y_train','X_test','y_test','X_val','y_val'])

    X_train=[]
    y_train = []
    X_test=[]
    y_test = []
    X_val=[]
    y_val = []

    # X_train=np.empty((num_training,48,48,1))
    # y_train = np.empty((num_training,1))
    # X_test=np.empty((num_test,48,48,1))
    # y_test = np.empty((num_test,1))
    # X_val=np.empty((num_val,48,48,1))
    # y_val = np.empty((num_val,1))

    for i in range(0,num_training):
        with Image.open(train_dir + str(i+1) + '.jpg').convert("L") as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 1))

            if(i < (num_val)):
                X_val.append(im_arr)
                y_val.append(labels[i])
            else:
                X_train.append(im_arr)
                y_train.append(labels[i])

#    print(X_train)
    for i in range(0,num_test):
        with Image.open(test_dir + str(i+28709+1) + '.jpg').convert("L") as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 1))
            X_test.append(im_arr)
            y_test.append(labels[i+28709])

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }

def get_FER2013_data_normalisation(num_training = 28709,num_test = 3589,num_val = 4000):

    if(num_val > num_training*0.2):
        print("num_val size too large, please insert something smaller")

    directory = 'datasets/public'
    train_dir = directory + '/Train/'
    test_dir = directory + '/Test/'

    labels = np.loadtxt(directory + '/labels_public.txt',skiprows=1,delimiter=',',usecols=1,dtype='int')

    X_train=[]
    y_train = []
    X_test=[]
    y_test = []
    X_val=[]
    y_val = []

    for i in range(0,num_training):
        with Image.open(train_dir + str(i+1) + '.jpg').convert("L") as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 1))

            if(i < (num_val)):
                X_val.append(im_arr)
                y_val.append(labels[i])
            else:
                X_train.append(im_arr)
                y_train.append(labels[i])

#    print(X_train)
    for i in range(0,num_test):
        with Image.open(test_dir + str(i+28709+1) + '.jpg').convert("L") as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 1))
            X_test.append(im_arr)
            y_test.append(labels[i+28709])

    mean_image = np.mean(X_train, axis=0, dtype=int)
    X_train = np.asarray(X_train) - mean_image
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test) - mean_image
    y_test = np.asarray(y_test)
    X_val = np.asarray(X_val) - mean_image
    y_val = np.asarray(y_val)

    # print(X_train)

    # Package data into a dictionary
    return {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
