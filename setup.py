# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:48:36 2019

@author: GU389021
"""

from keras.datasets import mnist
from keras.utils import np_utils 
from config import num_classes,height, width , depth



def load_dataset():
    global height, width , depth 
    (train_features,train_labels), (test_features,test_labels) = mnist.load_data()
    height, width, depth = train_features.shape[1], train_features.shape[2], 1 

    train_features = train_features.reshape(train_features.shape[0],height,width, depth)

    test_features = test_features.reshape(test_features.shape[0],height,width, depth)
    train_features = train_features.astype('float32')
    test_features = test_features.astype('float32')

    train_features/= 255 
    test_features /= 255 

    train_labels = np_utils.to_categorical(train_labels, num_classes)
    encoded_test_labels = np_utils.to_categorical(test_labels, num_classes)
    
    return ((train_features,train_labels),(test_features,test_labels),encoded_test_labels)


