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


def show_heatmap(c_matrix):
    sns.heatmap(c_matrix.T,annot = True,fmt = 'd',cbar = True)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

def save_plot(Wrong_digits_idx,out_df,test_features):
    plt.ioff()
    i = 1
    while True:
        idx_list = Wrong_digits_idx[9*i:9*(i+1)]
         
        if len(idx_list)==0:
            break
        fig = plt.figure()
        fig.suptitle("Actual(A) vs Predicted(P)", fontsize=16)
        for idx,idx_value in enumerate(idx_list):
            plt.subplot(3,3,idx+1)
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
        