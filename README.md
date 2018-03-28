# ConvolutionalNeuralNetwork

Implemented neural networks with convolutional, pooling and the dropout layers.
Dataset: MNIST - digits 0 to 9
Data can be downloaded using ('download mnist.sh').
The MNIST dataset has 60,000 images which may be too large for batch gradient descent. Therefore,
train with merely 6000 samples and test with 1000 samples.

## Neural network with one convolutional layer

Use 32 filters in the convolutional layer, each of them should be of size 3  3 patches with stride 1.
This layer is fully connected to the second hidden layer.

## Neural network with one convolutional layer and one pooling layer

Based on the network from the previous section, add a max pooling layer after the convolutional layer.
The pooling is 2  2 with stride 1.
This layer is fully connected to the second hidden layer.

## Introduing dropout to your network
Based on the network from the previous section, add the dropout technique for updating the weights.
The dropout rate set to 50%
