#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 16:24:40 2018

@author: michaelscott
"""

input_dict = pickle.load(open("datasets/public/image_dict.txt", "rb"))
for key in input_dict:
    print(input_dict[key])