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
# from fcnet import FullyConnectedNet
# from utils.solver import Solver
from utils.data_utils import get_CIFAR10_data, get_FER2013_data
import matplotlib.pyplot as plt
import tensorflow as tf


tf.logging.set_verbosity(tf.logging.INFO)

#Define the tensorflow model
def cnn_model(features, labels, mode):

    beta = 0.000001
    p = 0.0
    learn_rate = 0.01
    input_layer = tf.reshape(features["x"],[-1,48,48,1])
    input_layer = tf.cast(input_layer, tf.float32)

    # Regularise weights
    # reg1 = tf.contrib.layers.l2_regularizer(scale=beta)

    # First convolutional layer
    conv1 = tf.layers.conv2d(inputs = input_layer, filters = 64, kernel_size = [3,3], padding = "same",
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta),
                             activation = tf.nn.relu)
    # First pooling layer
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2],strides=2)
    # First dropout
    dropout1 = tf.layers.dropout(inputs = pool1, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
    # First BN
    bn1 = tf.layers.batch_normalization(dropout1,training = mode == tf.estimator.ModeKeys.TRAIN)

    # Second convolutional
    conv2 = tf.layers.conv2d(inputs=bn1, filters = 128, kernel_size = [5,5], padding = "same",
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta),
                             activation = tf.nn.relu)
    # second pooling
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2],strides=2)
    # Second dropout
    dropout2 = tf.layers.dropout(inputs = pool2, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
    # First BN
    bn2 = tf.layers.batch_normalization(dropout2,training = mode == tf.estimator.ModeKeys.TRAIN)

    # Third convolutional
    conv3 = tf.layers.conv2d(inputs=bn2, filters = 512, kernel_size = [3,3], padding = "same",
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta),
                             activation = tf.nn.relu)
    # third pooling
    pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2],strides=2)
    # Third dropout
    dropout3 = tf.layers.dropout(inputs = pool3, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
    # First BN
    bn3 = tf.layers.batch_normalization(dropout3,training = mode == tf.estimator.ModeKeys.TRAIN)

    # Third convolutional
    conv4 = tf.layers.conv2d(inputs=bn3, filters = 512, kernel_size = [3,3], padding = "same",
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta),
                             activation = tf.nn.relu)
    # fourth pooling
    pool4 = tf.layers.max_pooling2d(inputs = conv4, pool_size = [2,2],strides=2)
    # Third dropout
    dropout4 = tf.layers.dropout(inputs = pool4, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
    # First BN
    bn4 = tf.layers.batch_normalization(dropout4,training = mode == tf.estimator.ModeKeys.TRAIN)

    # ############ FULLY CONNECTED LAYERS

    #  First hidden
    pool5_flat = tf.reshape(bn4, [-1,3*3*512])
    dense1 = tf.layers.dense(inputs = pool5_flat, units = 256,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta),
                             activation = tf.nn.relu)
    # Hidden First dropout
    dropout5 = tf.layers.dropout(inputs = dense1, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)
    #  Second hidden
    dense2 = tf.layers.dense(inputs = dropout5, units = 512,
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=beta),
                             activation = tf.nn.relu)
    # Hidden First dropout
    # dropout6 = tf.layers.dropout(inputs = dense2, rate = p, training = mode == tf.estimator.ModeKeys.TRAIN)

    # Final logits layer
    logits = tf.layers.dense(inputs = dense2, units = 7)
    logits += 0.000001
    # print(logits)
    # Predictions for PREDICT and EVAL mode
    predictions = {"classes": tf.argmax(input=logits, axis=1),
                   "probabilities": (tf.nn.softmax(logits, name = "softmax_tensor"))
                   }
    accuracy = {"accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"],name="accuracy")}

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(labels, tf.int32),logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimise = tf.train.GradientDescentOptimizer(learning_rate = learn_rate)

        train = optimise.minimize(loss=loss, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train)

    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(labels=labels,predictions=predictions["classes"]),
            "recall": tf.metrics.recall(labels=labels,predictions=predictions["classes"]),
            "precision": tf.metrics.precision(labels=labels,predictions=predictions["classes"])
            }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops = eval_metric_ops)
