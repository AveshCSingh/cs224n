#!/usr/bin/env python

import numpy as np
import random

from math import sqrt
from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function

    Implement a function that normalizes each row of a matrix to have
    unit length.
    """

    ### YOUR CODE HERE
    norm_row = lambda r : r / sqrt(np.sum(r**2))
    x = np.apply_along_axis(norm_row, 1, x)
    ### END YOUR CODE

    return x


def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    print x
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print ""

def avesh_test():
    predicted = np.array([0.25, 0.3, 0.5, 0.7, 0.92]) 


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """
    ### YOUR CODE HERE
    outputVectors = np.matrix(outputVectors)
    # We transpose predicted because it should be a column vector
    predicted = np.matrix(predicted).transpose()
    # y_hat is a column vector of softmax predictions, should have size w.
    y_hat = (softmax((outputVectors * predicted).transpose())).transpose()
    y = np.zeros_like(y_hat)
    y[target] = 1

    # Forward Propagation
    cost = -1*np.log(y_hat[target, 0])

    # Compute gradient with respect to the predicted word vector
    # Unlike in my written solution, outputVectors is k x w, so no need to transpose

    gradPred = outputVectors.transpose() * (y_hat - y)

    # Compute gradient with respect to all other word vectors
    grad = (y_hat - y) * (predicted.transpose())
    ### END YOUR CODE

    return cost, gradPred, grad


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in xrange(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


@profile
def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    # Sampling of indices is done for you. Do not modify this if you
    # wish to match the autograder and receive points!
    indices = [target]
    indices.extend(getNegativeSamples(target, dataset, K))

    ### YOUR CODE HERE
    outputVectors = np.matrix(outputVectors)
    u_o = outputVectors[target]
    predicted = np.matrix(predicted).transpose()
    # Forward propagation
    cost_a = -1*np.log(sigmoid(u_o * predicted))[0,0]
    intermediate = np.log(sigmoid(-1*outputVectors[indices] * predicted))
    cost = cost_a - np.sum(intermediate)

    # Compute gradient with respect to the predicted word vector
    grad_pred_a = u_o.transpose() * (sigmoid(u_o * predicted - 1)[0,0])
    grad_pred_b = outputVectors[indices].transpose() * (sigmoid(-1*outputVectors[indices]*predicted)-1)

    gradPred = grad_pred_a - grad_pred_b
    # Compute gradient with respect to all other word vectors
    grad = (1-sigmoid(-1*outputVectors * predicted))*(predicted.transpose())
    grad_w_o = predicted*(sigmoid(u_o*predicted) - 1)
    grad[target] = grad_w_o.transpose()
    ### END YOUR CODE

    return cost, gradPred, grad


def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
             dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """ Skip-gram model in word2vec

    Implement the skip-gram model in this function.

    Arguments:
    currrentWord -- a string of the current center word
    C -- integer, context size
    contextWords -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    inputVectors -- "input" word vectors (as rows) for all tokens
    outputVectors -- "output" word vectors (as rows) for all tokens
    word2vecCostAndGradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """
    c = tokens[currentWord]

        #    result_softmax = softmaxCostAndGradient(inputVectors[C], tokens[currentWord], outputVectors, dataset)
        #   result_neg = negSamplingCostAndGradient(inputVectors[C], tokens[currentWord], outputVectors, dataset)

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    gradPred = np.zeros((gradIn.shape[1], 1))
    
    for j in range(-C, C):
        inputVector = inputVectors[c]
        cost_j, gradPred_j, grad_j = word2vecCostAndGradient(inputVector, tokens[currentWord], outputVectors, dataset)
        cost += cost_j
        gradPred += gradPred_j
        gradOut += grad_j

    gradIn[c] = gradPred.transpose()
    ### END YOUR CODE
    return cost, gradIn, gradOut


def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
         dataset, word2vecCostAndGradient=softmaxCostAndGradient):
    """CBOW model in word2vec

    Implement the continuous bag-of-words model in this function.

    Arguments/Return specifications: same as the skip-gram model

    Extra credit: Implementing CBOW is optional, but the gradient
    derivations are not. If you decide not to implement CBOW, remove
    the NotImplementedError.
    """

    cost = 0.0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut


#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
        skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    #print "\n==== Gradient check for CBOW      ===="
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #    cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
    #    dummy_vectors)
    #gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
    #    cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
    #    dummy_vectors)

    print "\n=== Results ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print skipgram("c", 1, ["a", "b"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    print cbow("a", 2, ["a", "b", "a", "c"],
        dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset,
        negSamplingCostAndGradient)


if __name__ == "__main__":
    test_normalize_rows()
    avesh_test()
    test_word2vec()
