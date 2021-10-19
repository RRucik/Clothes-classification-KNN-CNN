# Clothes classification using k-Nearest Neighbour and Convolutional Neural Networks
## Introduction
The task was to make an algorithm that is capable of classifying clothes images. 

Dataset source: [fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)

It is a dataset where training set consist of 60000 examples and test set of 10000 examples.
I made a KNN algorithm that is capable of predicting clothes classes with an accuracy of 73.52%, and a
Convolution Neural Network that is capable of predicting clothes classes with an accuracy of 92.38%

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
Our model summary looks like this:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
batch_normalization (BatchNo (None, 26, 26, 32)        128       
_________________________________________________________________
activation (Activation)      (None, 26, 26, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 32)        9248      
_________________________________________________________________
batch_normalization_1 (Batch (None, 24, 24, 32)        128       
_________________________________________________________________
activation_1 (Activation)    (None, 24, 24, 32)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 32)        0         
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0         
_________________________________________________________________
dense (Dense)                (None, 256)               1179904   
_________________________________________________________________
dropout (Dropout)            (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2570      
=================================================================
Total params: 1,192,298
Trainable params: 1,192,170
Non-trainable params: 128
_________________________________________________________________
```

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
Also, we want to rescalate the image so that each pixel lies in interval [0, 1] instead of [0, 255], so we simply divide our sets 
by 255.

```
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train = X_train / 255.0
X_test = X_test / 255.0
```


## Results

| Used method | Accuracy | Accuracy in benchmark
| --- | --- | --- |
| KNN | 73.52% | 84.7%
| CNN | 92.38% | 87.6%

### KNN output
```
Model accuracy: 0.7352000000000001 K value for lowest error: 40
```

### CNN output
```
Epoch 1/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.4178 - accuracy: 0.8516
Epoch 2/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.2664 - accuracy: 0.9009
Epoch 3/10
1875/1875 [==============================] - 67s 36ms/step - loss: 0.2216 - accuracy: 0.9181
Epoch 4/10
1875/1875 [==============================] - 67s 36ms/step - loss: 0.1882 - accuracy: 0.9307
Epoch 5/10
1875/1875 [==============================] - 69s 37ms/step - loss: 0.1597 - accuracy: 0.9403
Epoch 6/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.1363 - accuracy: 0.9492
Epoch 7/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.1130 - accuracy: 0.9588
Epoch 8/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.0952 - accuracy: 0.9639
Epoch 9/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.0805 - accuracy: 0.9700
Epoch 10/10
1875/1875 [==============================] - 68s 36ms/step - loss: 0.0680 - accuracy: 0.9743
313/313 - 1s - loss: 0.2907 - accuracy: 0.9238
```

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
of the methods that were presented in method section. The result that you will get might not be equal to mine, but it should be 
relatively close.

Source of inspiration for CNN:
https://www.youtube.com/watch?v=cAICT4Al5Ow
