#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 13:45:49 2018

@author: michaelscott
"""

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

from tensorflow.python.client import device_lib
print (device_lib.list_local_devices())
print("GPUs: ",get_available_gpus())
