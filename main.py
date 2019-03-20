# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:27:59 2019

@author: GU389021
"""
from setup import load_dataset,get_image_dimensions
from config import filters1,filters2,dropout_prob,hidden_size,num_classes,tensorboard_dir,batch_size,num_epochs,model_path
from model import model_design
from train import train_model,evalute_model
from predict import predictions,show_heatmap,show_plot
    

if __name__ =='__main__':
    (train_features,train_labels),(test_features,test_labels),encoded_test_labels= load_dataset()
    height,width,depth = get_image_dimensions()

    model,tensorboard = model_design(height=height,width= width,depth=depth,filters1=filters1,
                                 dropout_prob=dropout_prob,filters2=filters2,hidden_size=hidden_size,
                                 num_classes=num_classes,tensorboard_dir=tensorboard_dir)


    model_data = train_model(model,train_features,train_labels,batch_size,num_epochs,tensorboard)
    
    evalute_model(model,model_path,test_features,encoded_test_labels)

    c_matrix,Wrong_digits_idx,out_df = predictions(model_path,test_features,test_labels)
    
    show_heatmap(c_matrix)

