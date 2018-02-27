from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import scipy.misc
import platform
import PIL as pillow
from PIL import Image


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
    
    if(num_val < 4000):
        print("num_val size too small, please insert something bigger")
    
    if(num_val > num_training*0.2):
        print("num_val size too large, please insert something smaller")
    
    directory = 'datasets/public'
    train_dir = directory + '/Train/'
    test_dir = directory + '/Test/'
    
    labels = np.loadtxt(directory + '/labels_public.txt',skiprows=1,delimiter=',',usecols=1,dtype='int')
    
    print(labels)
    
#    data = dict.fromkeys(['X_train','y_train','X_test','y_test','X_val','y_val'])
    
    X_train=y_train=X_val=y_val=X_test=y_test = []
    

    
    for i in range(1,num_training+1):
        with Image.open(train_dir + str(i) + '.jpg').convert("L") as image:
#            print(image.shape)
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 1))
#            print(im_arr)
            X_train.append(im_arr)
            y_train.append(labels[i])
    
    for i in range(1,num_test+1):
        with Image.open(test_dir + str(i+28709) + '.jpg').convert("L") as image:
            im_arr = np.fromstring(image.tobytes(), dtype=np.uint8)
            im_arr = im_arr.reshape((image.size[1], image.size[0], 1))
            X_test = np.append(X_test, im_arr)
            y_test = np.append(y_test, labels[i+28708])
            
    
            
    # Subsample the data
    mask = list(range(num_training, num_training + num_val))
#    print("mask length ", len(mask))
    X_val = X_train[mask[0]:mask[-1]]
#    print("x_val ", X_val)
    y_val = y_train[mask[0]:mask[-1]]
    
    mask = list(range(num_training))
    X_train = X_train[mask[0]:mask[-1]]
    y_train = y_train[mask[0]:mask[-1]]
    
    out = {
      'X_train': X_train, 'y_train': y_train,
      'X_val': X_val, 'y_val': y_val,
      'X_test': X_test, 'y_test': y_test,
    }
    
#    print(out)
    
    # Package data into a dictionary
    return out
    
    
image_dict = get_FER2013_data()  
f = open('Datasets/public/image_dict.txt','wb')
pickle.dump(image_dict,f)
f.close()



    
    
    