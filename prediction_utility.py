# -*- coding: utf-8 -*-

from predict import unlabelled_predictions

## model relative or absolute path both will work
model_path = 'mnist.h5'

## more than one image
## images relative or absolute path both will work
list_image_path = ['images/img_3.jpg','images/img_8.jpg']
print(unlabelled_predictions(model_path, list_image_path))

## only one image
list_image_path = ['images/img_9.jpg']
print(unlabelled_predictions(model_path, list_image_path))
