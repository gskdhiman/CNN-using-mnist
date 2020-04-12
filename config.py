# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:46:52 2019

@author: GU389021
"""

import os 
#filters in first conv2d
filters1 = 128
#filters in second conv2d
filters2 = 256
dropout_prob = 0.3
hidden_size = 200 
num_classes = 10
batch_size = 256
num_epochs = 20

model_path = os.path.join(os.getcwd(),'mnist.h5')
tensorboard_dir = os.getcwd()

output = os.path.join(os.getcwd(),'output')

if not os.path.exists(output):
    os.makedirs(output)