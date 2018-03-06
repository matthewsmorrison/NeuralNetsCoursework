import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

import numpy as np
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data
from src.utils.data_utils import get_FER2013_data


import pickle
import sys
import traceback

def test_fer_model(img_folder, model='model.pkl'):
    f = open(model, "rb")
    model = pickle.load(f)
    preds = None
    i = 1
    while(True):
        try:
            Image.open(img_folder + str(i) + ".jpg")
        except FileNotFoundError as e:
            print(str(i-1) + " images found" )
            break
        i+= 1
    return preds


def save_fer_model():
    data_dict = get_FER2013_data()
    model = FullyConnectedNet([200],input_dim = 48*48*1,
                              dropout=0, reg=0.4,
                              dtype=np.float64, seed=237)
    number_epochs = 20
    solver = Solver(model,data_dict,optim_config={'learning_rate':1e-4},
                    lr_decay=1,num_epochs=number_epochs,batch_size=200,
                    print_every=5000,num_train_samples=40000)
    results = solver.train()
    f = open("model.pkl", "wb")
    pickle.dump(model, f)
    print("model saved")
    
#save_fer_model()
print(test_fer_model("/home/vincent/Documents/imperial/machine learning/Coursework/NeuralNetsCoursework/datasets/public/Train/"))
