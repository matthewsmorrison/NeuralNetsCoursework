import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

import numpy as np
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver, confusion_matrix, performanceMetrics
from src.utils.data_utils import get_CIFAR10_data
from src.utils.data_utils import get_FER2013_data_normalisation


import pickle

import glob

def test_fer_model(img_folder, model='model.pkl'):
    f = open(model, "rb")
    model = pickle.load(f)
    print("model loaded")
    preds = None
    i = 1
    X_test = []
    print("getting images")
    images = glob.glob(img_folder + "/*.jpg")
    images.sort()
    for image in images:
        img = Image.open(image).convert("L")
        im_arr = np.fromstring(img.tobytes(), dtype=np.uint8)
        im_arr = im_arr.reshape((img.size[1], img.size[0], 1))
        im_arr = np.asarray(im_arr) - model.mean_image
        img.close()
        X_test.append(im_arr)
        i+= 1
    print("got images")
    X_test = np.array(X_test)
    scores = model.loss(X_test)
    y_pred = np.argmax(scores, axis=1)

    return y_pred



