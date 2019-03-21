# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:27:59 2019

@author: GU389021
"""
from setup import load_dataset,get_image_dimensions
from config import filters1,filters2,dropout_prob,hidden_size,num_classes,tensorboard_dir,batch_size,num_epochs,model_path
from model import model_design
from train import train_model,evalute_model
from predict import predictions,show_heatmap,save_plot,show_acc_loss_plot

want_to_train = True    

if __name__ =='__main__':
    # load the training set into train_features,train_labels and test set into test_features,encoded_test_labels
    # test_labels are not encoded
    (train_features,train_labels),(test_features,test_labels),encoded_test_labels= load_dataset()
    
    # will return you the height, width , depth of the image
    height,width,depth = get_image_dimensions()
    
    # if you want to train 
    if want_to_train:
        model = model_design(height=height,width= width,depth=depth,filters1=filters1,
                         dropout_prob=dropout_prob,filters2=filters2,hidden_size=hidden_size,
                         num_classes=num_classes,tensorboard_dir=tensorboard_dir)

        model_data = train_model(model,model_path,train_features,train_labels,batch_size,
                                 num_epochs,tensorboard_dir)    
    
    #evaluate the saved model    
    evalute_model(model_path,test_features,encoded_test_labels)
    
    #get the confusion matrix and incorrectly predicted images index and the dataframe containing all this info
    c_matrix,Wrong_digits_idx,out_df = predictions(model_path,test_features,test_labels)
    
    # see the result in a heatmap
    show_heatmap(c_matrix)
    
    # save the misclassified images 4*4 subplots in the output(see config.py) folder
    save_plot(Wrong_digits_idx,out_df,test_features)
    
    #It will show you the loss and accuracy plots
    show_acc_loss_plot(model_data)
    

