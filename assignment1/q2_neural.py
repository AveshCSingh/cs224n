#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    data = np.matrix(data)
    W1 = np.matrix(W1)
    b1 = np.matrix(b1)
    W2 = np.matrix(W2)
    b2 = np.matrix(b2)

    z1 = data * W1 + b1
    h = sigmoid(z1)

    z2 = h * W2 + b2
    y_pred = softmax(z2)
    cost_per_label = np.sum(-1*np.multiply(labels, np.log(y_pred)), axis=1)
    cost = np.mean(cost_per_label)

    ### END YOUR CODE

    ### YOUR CODE HERE: backward propagation
    Dy = labels.shape[-1]
    Dx = data.shape[-1]
    H = h.shape[-1]
    n = labels.shape[0]

    gradb2_per_label = y_pred - labels
    gradb2 = np.mean(gradb2_per_label, axis=0)

    gradW2_per_label = h.transpose()*(y_pred - labels)
    gradW2 = gradW2_per_label / n

    # sigmoid_grad expects the argument to be sigmoid(x), so we pass in h not z1
    gradb1_per_label = np.multiply((y_pred - labels) * W2.transpose(), sigmoid_grad(h))
    gradb1 = np.mean(gradb1_per_label, axis=0)

    gradW1_per_label = data.transpose() * np.multiply(sigmoid_grad(h), (y_pred - labels)*W2.transpose())
    gradW1 = gradW1_per_label / n

    gradb2 = np.array(gradb2)
    gradW2 = np.array(gradW2)
    gradb1 = np.array(gradb1)
    gradW1 = np.array(gradW1)
    ### END YOUR CODE

    ### Stack gradients (do not modify)
    
    #W1: 0-49 
    #b1: 50-54
    #W2: 55-104
    #b2: 105-114
    
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    # raise NotImplementedError
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
