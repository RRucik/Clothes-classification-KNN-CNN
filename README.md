# Clothes classification using k-Nearest Neighbour and Convolution Neural Networks
## Introduction
The task was to make an algorithm that is capable of clasificating clothes images. 
Dataset source: [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
It is a dataset where training set consist of 60000 examples and test set of 10000 examples.
I made a KNN algorithm that is capable of predicting clothes classes with an accuracy of 73.5%, and a
Convolution Neural Network that is capable of predicting clothes classes with an accuracy of 92.32%

## Methods
The whole code is placed in the folder called scripts

### K-Nearest Neighbour
Script knn.py consists of several methods:

hamming_distance - Creates a hamming distance matrix for two matrices passed as parameters

sort_train_labels_knn - Sorts classes by hamming distance matrix

p_y_x_knn - Returns probability matrix of p(y|x) for each class from test set given some parameter k

classification_error - Calculates classification error

model_selection_knn - Returns model accuracy, best parameter k and array of errors for different k, given training and test sets
and an array of k values

test_knn - Loads dataset and runs model_selection_knn for some k_values. 


### Convolutional Neural Networks
In order to create this neural network, I am using tensorflow and keras.
Script neural_network.py consist only of method test_cnn. This method runs our program.
Our neural_network is made by using following layers:



Each of these functions is described bellow:

Convolution2D - creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs.

BatchNormalization - normalizes and scales inputs.

Activation - applies an activation function to an output

MaxPooling2D - downsamples the input representation by taking the maximum value over the window defined by pool_size

Flatten - flattens the input

Dense - your regular densely-connected NN layer

Dropout - the dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, 
which helps prevent overfitting


In order to be able to use Convolution2D on our dataset, first I need to change our training and test shape to 4 dimentional.
Now training test will have shape (60000, 28, 28, 1) and test set (10000, 28, 28, 1). 28 is the picture size, which is
equal to 28x28 pixels and 1 is the number of channels, which is equal to that number because our pictures are in grayscale.


## Results
Results for KNN:

Results for CNN:

## Usage
### Running KNN
In order to run KNN, first we need to import mnist_reader from folder utils, and numpy. After that, we need to run method
test_knn. Our dataset will be loaded automaticly after running this method. Program might need to run at least 30 
minutes before returning the result. In order to achieve similar results that I did, you should set k_values in test_knn 
method to these: [10, 20, 30, 40, 50]. In the output, we will see the accuracy of our model and K value for which accuracy was the best.

### Running CNN
In order to run CNN, first we need to import keras and tensorflow. After that, we need to run method test_CNN.
Our dataset will be loaded automaticly after running this method. When program is running, we will be able to see how much time
is left to finish calculating result for each epoch, epoch accuracy and loss. After program finishes running, we will see
overall loss and accuracy achieved by our neural network. If you want to get the same output that I did, do not change parameters 
of the methods that were presented in method section.
