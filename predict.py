# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:45:52 2019

@author: GU389021
"""

from keras.models import load_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix #,cohen_kappa_score
import seaborn as sns
import keyboard
import os
from config import output
from PIL import Image

def predictions(model_path,test_features,test_labels):
    model = load_model(model_path)
    predictions = model.predict(test_features)
    
    predicted_digits = [np.argmax(prediction) for prediction in predictions]
    actual_labels = list(test_labels.astype(int))
        
    out_df = pd.DataFrame({'actual_label':actual_labels,
                           'predicted':predicted_digits})
    Wrong_digits_idx = list(np.where(out_df['actual_label'] != out_df['predicted'])[0])
    c_matrix = confusion_matrix(actual_labels,predicted_digits)
    return(c_matrix,Wrong_digits_idx,out_df)

def unlabelled_predictions(model_path: str,list_image_path: list)-> list:
    '''
    Parameters
    ----------
    model_path : h5
        The path for the trained model and as of now it is given in the project 
        as model.h5 file
    image_path : jpg.jpeg
        The list of relative path of the image which you want to predict.
    Returns
    -------
    predicted_digit : str 
        returns the digit number which the image contains but in str format so that this functionality can
        be used later for some project.
    '''
    ## no need already type checking
    if isinstance(list_image_path,str):
         list_image_path = [list_image_path]       
    # image_path = 'images\img_49.jpg'
    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print('The path given for the model is not correct')
    predicted_digits = []
    for image_path in list_image_path:
        # Open the image form the given path 
        image = Image.open(image_path)
        image = np.array(image)
        ## add the channel dim 
        ## before 28,28 now 28,28,1
        image = np.expand_dims(image,2)
        ## add the batch dim
        ## before (28,28,1) after (1,28,28,1)
        image = np.expand_dims(image,0)
        predicted_digits.append(str(np.argmax(model.predict(image))))
    return predicted_digits

def show_heatmap(c_matrix):
    sns.heatmap(c_matrix.T,annot = True,fmt = 'd',cbar = True)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def save_plot(Wrong_digits_idx,out_df,test_features):
    plt.ioff()
    plt.rcParams['font.size'] = 6
    i = 1
    while True:
        idx_list = Wrong_digits_idx[16*i:16*(i+1)]
         
        if len(idx_list)==0:
            break
        fig = plt.figure()
        fig.suptitle("Actual(A) vs Predicted(P)", fontsize=7)
        for idx,idx_value in enumerate(idx_list):
            plt.subplot(4,4,idx+1)
            
            plt.imshow(test_features[idx_value].reshape(28,28), cmap='gray', interpolation='none')
            digits_compare = out_df.iloc[idx_value].tolist()
            plt.title('A:'+str(digits_compare[0])+ ' but ' +'P:'+str(digits_compare[1]))
            plt.xticks([])
            plt.yticks([])
        plt.close(fig)
        fig.savefig(os.path.join(output,'Wrong_predictions'+str(i)+'.jpg'))
        i = i+1
        
#        while True:
#            try:
#                if keyboard.is_pressed(' '): 
#                    break
#                else:
#                    pass
#            except:
#                break

def show_acc_loss_plot(model_data):
    plt.subplot(2,1,1)
    plt.plot(model_data.history['acc'])
    plt.plot(model_data.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    
    plt.subplot(212)  
    plt.plot(model_data.history['loss'])
    plt.plot(model_data.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
        