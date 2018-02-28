#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 17:42:44 2018

@author: michaelscott
"""
from __future__ import absolute_import, division, print_function
from builtins import range
from six.moves import cPickle as pickle
import numpy as np
import os
import scipy.misc
import platform
import PIL as pillow
from PIL import Image
from src.fcnet import FullyConnectedNet
from src.utils.solver import Solver
from src.utils.data_utils import get_CIFAR10_data, get_FER2013_data
import matplotlib.pyplot as plt
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

#Define the tensorflow model
def cnn_model(features, mode, labels):
    
#    define the input layer
#    if mode==Train:
#        X = features["X_train"]
#        y = features["y_train"]
#    else:
#        X = features["X_test"]
#        y = features["y_test"]
    p = 0.4
    learn_rate = 0.0001
    input_layer = tf.reshape(features["x"],[-1,48,48,1])
    input_layer = tf.cast(input_layer, tf.float32)

# First convolutional layer
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 64, kernel_size = [3,3], padding = "same", activation = tf.nn.relu)
# First pooling layer
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2],strides=2)
# First dropout
    dropout1 = tf.layers.dropout(inputs = pool1, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)


# Second convolutional
    conv2 = tf.layers.conv2d(inputs=dropout1, filters = 128, kernel_size = [5,5], padding = "same", activation = tf.nn.relu)
#second pooling
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2],strides=2)
# Second dropout
    dropout2 = tf.layers.dropout(inputs = pool2, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)


# Third convolutional
    conv3 = tf.layers.conv2d(inputs=dropout2, filters = 512, kernel_size = [3,3], padding = "same", activation = tf.nn.relu) 
#third pooling
    pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2],strides=2)
# Third dropout
    dropout3 = tf.layers.dropout(inputs = pool3, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)


# Third convolutional
    conv4 = tf.layers.conv2d(inputs=dropout3, filters = 512, kernel_size = [3,3], padding = "same", activation = tf.nn.relu) 
# fourth pooling
    pool4 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2,2],strides=2)
# Third dropout
    dropout4 = tf.layers.dropout(inputs = pool4, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)

     
# ############ FULLY CONNECTED LAYERS
    
#  First hidden
    pool5_flat = tf.reshape(dropout4, [-1,3*3*512])
    dense1 = tf.layers.dense(inputs = pool5_flat, units = 256, activation = tf.nn.relu)
# Hidden First dropout
    dropout5 = tf.layers.dropout(inputs = dense1, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
#  Second hidden
    dense2 = tf.layers.dense(inputs = dropout5, units = 512, activation = tf.nn.relu)
# Hidden First dropout
    dropout6 = tf.layers.dropout(inputs = dense2, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
    
# Final logits layer
    logits = tf.layers.dense(inputs = dropout6, units = 10)

# Predictions
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": tf.nn.softmax(logits, name = "softmax_tensor")
                   }
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    
    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32),logits=logits)
    
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimise = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)
        train = optimise.minimize(loss=loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train)
    
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops = eval_metric_ops)





