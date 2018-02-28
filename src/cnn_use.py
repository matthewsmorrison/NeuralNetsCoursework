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
import platform
import PIL as pillow
from PIL import Image
#from src.fcnet import FullyConnectedNet
#from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data, get_FER2013_data
import matplotlib.pyplot as plt
import tensorflow as tf
from src.cnn import cnn_model

def main(unused_argv):
  
#   LOADING THE DATA
#    directory = 'datasets/public'
#    train_dir = directory + '/Train/'
#    test_dir = directory + '/Test/'
    
    
#train_sz = 5000
#test_sz = 2000
#
#    
#train_image_batch = tf.placeholder(dtype=tf.float64, shape = [None,48*48])
#train_labels_batch = tf.placeholder(dtype=tf.float64, shape = [None,])
#test_image_batch = tf.placeholder(dtype=tf.float64, shape = [None,48*48])
#test_labels_batch =  tf.placeholder(dtype=tf.float64, shape = [None,])
#
#with tf.Session() as sesh:
#    tf.initalize_all_variables().run()
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(coord=coord)
#    image_tensor = sesh.run([image])
    
    
################################################################################
    
    #bring in the FER2013 data
    #input_dict = pickle.load(open("datasets/public/image_dict_short.pkl", "rb"))
    data = get_FER2013_data() 
#    training_data = data["X_train"]
#    train_data = tf.contrib.learn.datasets.load_dataset(training_data)
#    train_labels = np.asarray(data["y_train"],dtype=np.int32)
#    test_data = np.asarray(data["X_test"]).test.images
#    test_labels = np.asarray(data["y_test"],dtype=np.int32)  
#    train_tensor, test_tensor, y_train, y_test =   get_FER2013_data_tensor(num_training = 5000,num_test = 1000) 
    
#    for key, value in data.items() :
#        print(key)
#    print(train_data)
#    Create the estimator
    classifier = tf.estimator.Estimator(model_fn = cnn_model,model_dir = "tmp/model")
#    
##    log the predictions
    log_tensors = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = log_tensors, every_n_iter = 128)
#    
##    Train the model
#    
    
#    a = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0], shape=[2,3],name='a')
#    b = tf.constant([1.0,2.0,3.0,4.0,5.0,6.0], shape=[3,2],name='b')
#    c=tf.matmul(a,b)
#    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#    print(sess.run(c))
    
    
    
    training_input = tf.estimator.inputs.numpy_input_fn(
            x={"x":data['X_train']},
            y = data['y_train'],
            batch_size = 128,
            num_epochs = 3,
            shuffle = True)
    
    classifier.train(
            input_fn = training_input,
            max_steps = 100,
            hooks = [logging_hook]      
            )
    
##    Model evaluation
    evaluation_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x":data['X_test']},
            y = data['y_test'],
            num_epochs = 1,
            shuffle = False          
            )
#    
#    train_spec = tf.estimator.TrainSpec(input_fn=training_input, max_steps=10000)
#    eval_spec = tf.estimator.EvalSpec(input_fn=evaluation_fn)

#    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
    results = classifier.evaluate(input_fn = evaluation_fn)
    print(results)
            
if __name__=="__main__":
    tf.app.run()          