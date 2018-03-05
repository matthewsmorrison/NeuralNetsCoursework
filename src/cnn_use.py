#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 19:26:28 2018

@author: michaelscott
"""

from __future__ import absolute_import, division, print_function
from builtins import range
#from six.moves import cPickle as pickle
import numpy as np
import os
import scipy.misc
# import platforms
import pickle
import PIL as pillow
from PIL import Image
# from fcnet import FullyConnectedNet
# from utils.solver import Solver
from utils.data_utils import get_CIFAR10_data, get_FER2013_data, read_fer2013
import matplotlib.pyplot as plt
import tensorflow as tf
from cnn import cnn_model

def main(unused_argv):

#   LOADING THE DATA
#    directory = 'datasets/public'
#    train_dir = directory + '/Train/'
#    test_dir = directory + '/Test/'



################################################################################


#    Create the estimator
    classifier = tf.estimator.Estimator(model_fn = cnn_model,model_dir = "model_checkpoints_6classes6")
#

#
##    Train the model
#   Retrieve the data

    data = read_fer2013()

#   produce the input data stream
    epochs = 10
    training_input = tf.estimator.inputs.numpy_input_fn(
            x={"x":(data['X_train']-127.5)/127.5},
            y = data['y_train'],
            batch_size = 128,
            num_epochs = epochs,
            shuffle = True)
# Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=500)

#    Training phase
    classifier.train(
            input_fn = training_input,
            hooks = [logging_hook]
            )

##    Model evaluation
    evaluation_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":(data['X_val']-127.5)/127.5},
            y = data['y_val'],
            num_epochs = 1,
            shuffle = False
            )
#
   # train_spec = tf.estimator.TrainSpec(input_fn=training_input)
   # eval_spec = tf.estimator.EvalSpec(input_fn=evaluation_fn)

#    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)

    results = classifier.evaluate(input_fn = evaluation_fn)
    print(results)
    name = "src/results" + str(epochs) + ".p"
    pickle.dump(results,open(name,"wb"))



if __name__=="__main__":
    tf.app.run()
