'''
python 2.7
This file implements a multi layer neural network for a multiclass classifier

'''
import numpy as np
import datetime
import time
from load_dataset import mnist
import matplotlib.pyplot as plt
import pdb
import sys, ast


def relu(Z):
    '''
    computes relu activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0, Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ


def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z
    cache = {}
    cache["Z"] = Z
    return A, cache


def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    return dZ


def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs:
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []

    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    ### CODE HERE


    maxZ = np.max(Z, axis=0)
    expZ = np.exp(Z - maxZ)
    A = expZ / np.sum(np.exp(Z - np.max(Z, axis=0)), axis=0, keepdims=True)

    if Y.shape[0] != 0:
        loss = -np.sum(Y * np.log(A + 10 ** -6)) / A.shape[1]
    else :
        loss = []

    cache = {}
    cache["A"] = A
    return A, cache, loss


def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs:
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    ### CODE HERE
    A = cache["A"]
    dZ = A - Y
    return dZ


def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs:
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''
    np.random.seed(0)
    numLayers = len(net_dims)
    parameters = {}
    for l in range(numLayers - 1):
        parameters["W" + str(l + 1)] = np.random.normal(0, 2 / net_dims[l] ** 0.5, (net_dims[l + 1], net_dims[l]))
        parameters["b" + str(l + 1)] = np.random.normal(0, 2 / net_dims[l] ** 0.5, (net_dims[l + 1], 1))
    return parameters

def initialize_convlayer_weights():
    '''
    Initializes the weights of the convulational network

    Returns:
        dictionary of convulational parameters
    '''
    conv_parameters = {}

    conv_parameters["W"] = np.random.randn(3, 3, 1, 5)
    conv_parameters["b"] = np.random.randn(1, 1, 1, 5)
    conv_parameters["stride"] = 1
    conv_parameters["pad"] = 1
    conv_parameters["f"] = 3
    conv_parameters["ch"] = 5
    return conv_parameters


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values = (0,0))
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    s = np.multiply(a_slice_prev,W)
    Z = np.sum(s)
    Z = Z+b
    return Z


def convulation_forward(A_prev,conv_parameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    W = conv_parameters["W"]
    b = conv_parameters["b"]
    (f, f, n_C_prev, n_C) = W.shape

    stride = conv_parameters["stride"]
    pad = conv_parameters["pad"]

    n_H = int((n_H_prev - f + 2 * pad)/stride) + 1
    n_W = int((n_W_prev - f + 2 * pad)/stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev,W[:,:,:,c], b[:,:,:,c])

    cache = (A_prev, W, b, conv_parameters)
    return Z, cache

def convo_forward(X, parameters):
    Z, lin_cache = convulation_forward(X, parameters)
    cache = {}
    cache["lin_cache"] = lin_cache
    A, act_cache = relu(Z)
    cache["act_cache"] = act_cache
    return A, cache

def linear_forward(A, W, b):
    '''
    Input A propagates through the layer
    Z = WA + b is the output of this layer.

    Inputs:
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    ### CODE HERE
    Z = np.dot(W, A) + b
    cache = {}
    cache["A"] = A
    return Z, cache


def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs:
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b)
    if activation == "relu":
        A, act_cache = relu(Z)
    elif activation == "linear":
        A, act_cache = linear(Z)

    cache = {}
    cache["lin_cache"] = lin_cache
    cache["act_cache"] = act_cache
    return A, cache




def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
    caches.append(cache)
    return AL, caches




def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A = cache["A"]
    n, m = A.shape
    dA_prev = np.dot(W.T, dZ)
    dW = np.dot(dZ, A.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m

    ## CODE HERE
    return dA_prev, dW, db


def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs:
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = \
            layer_backward(dA, caches[l - 1], \
                           parameters["W" + str(l)], parameters["b" + str(l)], \
                           activation)
        activation = "relu"
    return dA,gradients



def convulational_backward(dZ, cache):

    (A_prev, W, b, conv_parameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    (f, f, n_C_prev, n_C) = W.shape

    stride = conv_parameters["stride"]
    pad = conv_parameters["pad"]
    (m, n_H, n_W, n_C) = dZ.shape

    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.zeros((1, 1, 1, n_C))

    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    for i in range(m):                       # loop over the training examples

        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]

        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f

                    a_slice = A_prev_pad[i, vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]

        dA_prev[i, :, :, :] = da_prev_pad[pad:-pad, pad:-pad, :]
    return dA_prev, dW, db


def convo_backward(dA, cache):
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]
    (A_prev, _, _, _) = lin_cache
    dA = np.reshape(dA, (A_prev.shape[0], 10, 10, 5))

    dZ = relu_der(dA, act_cache)
    dA_prev, dW, db = convulational_backward(dZ, lin_cache)
    return dA_prev, dW, db


def classify(X, parameters,conv_parameters):
    '''
    Network prediction for inputs X

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    ### CODE HERE
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions
    A0 = X
    # A0 = np.reshape(A0, (A0.shape[1], 10, 10, 1))
    A1, cache_convo = convo_forward(A0, conv_parameters)
    A1 = np.reshape(A1, (A1.shape[1] * A1.shape[2] * A1.shape[3] , A1.shape[0]))
    A, c1 = multi_layer_forward(A1, parameters)
    A, c2, l = softmax_cross_entropy_loss(A)
    YPred = np.argmax(A, axis=0)
    return np.reshape(YPred, (1, YPred.shape[0]))


def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.0):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    # alpha = learning_rate * (1 / (1 + decay_rate * epoch))
    alpha = learning_rate
    L = len(parameters) // 2
    ### CODE HERE
    for i in range(L):
        parameters["W" + str(i + 1)] -= alpha * gradients["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] -= alpha * gradients["db" + str(i + 1)]
    return parameters, alpha


def multi_layer_network(X, Y, net_dims, num_iterations=500, learning_rate=0.2, decay_rate=0.01):
    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent

    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    # print "in multi layer network"
    net_dims[0] = 10*10*5
    parameters = initialize_multilayer_weights(net_dims)
    conv_parameters = initialize_convlayer_weights()
    A0 = X
    # A0 = np.reshape(A0, (A0.shape[1], 10, 10, 1))
    costs = []
    n, m = Y.shape
    oneHotY = np.zeros((10, m))
    for i in range(m):
        oneHotY[int(Y[0, i]), i] = 1

    for ii in range(num_iterations):
        ### CODE HERE
        # Forward Prop
        # print"@iteration "+str(ii)
        forward_starttime = time.time()
        A1, cache_convo = convo_forward(A0, conv_parameters)
        forward_endtime = time.time()
        A1 = np.reshape(A1, (A1.shape[1] * A1.shape[2] * A1.shape[3], A1.shape[0]))
        # print A1.shape
        # print " forward convoluion pass for "+str(ii)+" done in "+str(forward_endtime - forward_starttime)  +" seconds, now calling multi_layer_forward"
        Z, caches = multi_layer_forward(A1, parameters)        ## call to multi_layer_forward to get activations
        ## call to softmax cross entropy loss
        # print " forward convoluion pass for " + str(ii) + " done "
        A, cache2, cost = softmax_cross_entropy_loss(Z, oneHotY)

        # Backward Prop
        dZ = softmax_cross_entropy_loss_der(oneHotY, cache2)
        # print " calling multi_layer_backward for "+ str(ii)
        dA,gradients = multi_layer_backward(dZ, caches, parameters)
        parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate)
        # print "multi_layer backward done now calling convulational_backward for " + str(ii)
        backward_starttime = time.time()
        _, dW, db = convo_backward(dA, cache_convo)
        # print dW
        backward_endtime = time.time()
        # print str(ii)+ " done! in "+str(backward_endtime - backward_starttime)+ " seconds."
        conv_parameters["W"] -= alpha * dW
        conv_parameters["b"] -= alpha * db
        ## call to softmax cross entropy loss der
        ## call to multi_layer_backward to get gradients
        ## call to update the parameters
        print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))
        # if ii % 10 == 0:
        #     costs.append(cost)
        costs.append(cost)
        # if ii % 10 == 0:
        #     print("Cost at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))

    return costs, parameters,conv_parameters


def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800]"
    The network will have the dimensions [784,800,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits

    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800,500]"
    The network will have the dimensions [784,800,500,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
    '''
    print "started at :"+ str(datetime.datetime.now().time())
    net_dims = ast.literal_eval(sys.argv[1])
    net_dims.append(10)  # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims))

    # getting the subset dataset from MNIST
    train_data, train_label, test_data, test_label = mnist(ntrain=6000, ntest=1000, digit_range=[0, 10])
    # initialize learning rate and num_iterations
    # print train_data.shape
    # print train_label.shape

    #
    # z1 = np.reshape(train_data, (train_data.shape[1] * train_data.shape[2] * train_data.shape[3], train_data.shape[0]))
    # print z1.shape
    learning_rate = 0.2
    num_iterations = 500

    costs, parameters,conv_parameters = multi_layer_network(train_data, train_label, net_dims, num_iterations=num_iterations,
                                            learning_rate=learning_rate)

    # compute the accuracy for training set and testing set
    # print "multi_layer_network done!"
    train_Pred = classify(train_data, parameters,conv_parameters)
    test_Pred = classify(test_data, parameters,conv_parameters)

    trAcc = np.mean((train_Pred == train_label)) * 100
    teAcc = np.mean((test_Pred == test_label)) * 100
    print("Accuracy for training set is {0:0.3f} %".format(trAcc))
    print("Accuracy for testing set is {0:0.3f} %".format(teAcc))

    ### CODE HERE to plot costs
    X = range(0, 500, 1)
    plt.plot(X,costs)
    plt.xlabel("iterations")
    plt.ylabel("Error")
    plt.title("Plot of train error vs  iterations")
    plt.show()
    print "ended at :" +str(datetime.datetime.now().time())


if __name__ == "__main__":
    main()