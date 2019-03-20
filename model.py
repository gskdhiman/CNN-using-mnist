# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 12:32:31 2019

@author: GU389021
"""
import os
from keras.models import Model
from keras.layers import Input, Dense 
from keras.layers import Conv2D, MaxPooling2D,Flatten,Dropout
from keras.callbacks import TensorBoard
    

def model_design(height,width,depth,filters1,
                 dropout_prob,filters2,hidden_size,
                 num_classes,tensorboard_dir):
    
    input_layer = Input(shape=(height,width,depth,))
    conv1= Conv2D(filters = filters1, kernel_size=(3, 3),activation='relu')(input_layer)
    maxpool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    dropout1 = Dropout(dropout_prob)(maxpool1)
    
    conv2= Conv2D(filters = filters2, kernel_size=(3, 3),activation='relu')(dropout1)
    maxpool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    dropout2 = Dropout(dropout_prob)(maxpool2)


    flatten = Flatten()(dropout2) 
    fc1 = Dense(hidden_size, activation='relu')(flatten) 
    fc2 = Dense(hidden_size, activation='relu')(fc1) 
    output = Dense(num_classes, activation='softmax')(fc2) 
    model = Model(inputs=input_layer, outputs=output) 

    model.compile(loss='categorical_crossentropy', 
              optimizer='nadam', 
              metrics=['accuracy']) 
    tensorboard = TensorBoard(log_dir=os.path.join(tensorboard_dir,'logs'),
                          histogram_freq=1,
                          write_graph=True)
    model.summary()
    model_data = model.fit(train_features, train_labels, 
                           batch_size=batch_size, epochs=num_epochs,
                           verbose=1,callbacks = [tensorboard],validation_split = 0.1) 
    return model,model_data,tensorboard