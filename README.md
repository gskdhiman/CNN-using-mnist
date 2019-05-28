
# <em> CNN using Mnist
#### The code presents a little different way of coding for mnist dataset. 

## Highlights of this project.

* Separate scripts of setup, model design, train and predict.
* Tensorboard to monitor loss.
* Intuitive way to understand the false postives.
* Heatmap for confusion matrix.
* Neat and clean (WIP).


## How to run:
<b>Note:</b> assuming you have the necessary libraries(python, pandas, numpy, keras, sklearn and etc.
* open cmd
* Go into the project directory 
* Run the following command

`python main.py`

This will run all the other scripts in the order defined in this script. <br>
You can have a look at this script first and then go into individual scripts.

The model will be stored in the current directory.

The sample code is able to achieve 99.18 accuracy without much effort and can be further improved.<br>
The sample model is also given (mnist.h5) to try without training. (toggle the flag want_to_train in main.py based on requirement).

<br>

The heatmap will show confusion matrix to see the output in a better form.<br>
The output folder will contain the misclassified images.
</em>
