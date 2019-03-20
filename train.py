# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:04:20 2019

@author: GU389021
"""
import os
from keras.callbacks import TensorBoard

def train_model(model,model_path,train_features,train_labels,batch_size,num_epochs,tensorboard_dir):
#    tensorboard = TensorBoard(log_dir=os.path.join(tensorboard_dir,'logs'),
#                          histogram_freq=1,
#                          write_graph=True)
    model_data = model.fit(train_features, train_labels, 
                           batch_size=batch_size, epochs=num_epochs,
                           verbose=1,validation_split = 0.1) 
    model.save(model_path)
    return model_data


def evalute_model(model,model_path,test_features,encoded_test_labels):
    from keras.models import load_model
    model = load_model(model_path)
    loss_acc_data = model.evaluate(test_features,encoded_test_labels , verbose=1)
    print(' The accuracy and loss of the system: ')
    print(dict(zip(model.metrics_names,loss_acc_data)))
    