import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
#matplotlib inline

def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    # np.exp exponential function imported from numpy
    return 1.0 / (1.0 + np.exp(-1.0 * z))


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Pick a reasonable size for validation data

    # ------------Initialize preprocess arrays----------------------#
    train_preprocess = np.zeros(shape=(50000, 784))
    validation_preprocess = np.zeros(shape=(10000, 784))
    test_preprocess = np.zeros(shape=(10000, 784))
    train_label_preprocess = np.zeros(shape=(50000,))
    validation_label_preprocess = np.zeros(shape=(10000,))
    test_label_preprocess = np.zeros(shape=(10000,))
    # ------------Initialize flag variables----------------------#
    train_len = 0
    validation_len = 0
    test_len = 0
    train_label_len = 0
    validation_label_len = 0
    # ------------Start to split the data set into 6 arrays-----------#
    for key in mat:
        # -----------when the set is training set--------------------#
        if "train" in key:
            label = key[-1]  # record the corresponding label
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)  # get the length of current training set
            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set

            # ---------------------adding data to training set-------------------------#
            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]
            train_len += tag_len

            train_label_preprocess[train_label_len:train_label_len + tag_len] = label
            train_label_len += tag_len

            # ---------------------adding data to validation set-------------------------#
            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]
            validation_len += 1000

            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label
            validation_label_len += 1000

            # ---------------------adding data to test set-------------------------#
        elif "test" in key:
            label = key[-1]
            tup = mat.get(key)
            sap = range(tup.shape[0])
            tup_perm = np.random.permutation(sap)
            tup_len = len(tup)
            test_label_preprocess[test_len:test_len + tup_len] = label
            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]
            test_len += tup_len
            # ---------------------Shuffle,double and normalize-------------------------#
    train_size = range(train_preprocess.shape[0])
    train_perm = np.random.permutation(train_size)
    train_data = train_preprocess[train_perm]
    train_data = np.double(train_data)
    train_data = train_data / 255.0
    train_label = train_label_preprocess[train_perm]

    validation_size = range(validation_preprocess.shape[0])
    vali_perm = np.random.permutation(validation_size)
    validation_data = validation_preprocess[vali_perm]
    validation_data = np.double(validation_data)
    validation_data = validation_data / 255.0
    validation_label = validation_label_preprocess[vali_perm]

    test_size = range(test_preprocess.shape[0])
    test_perm = np.random.permutation(test_size)
    test_data = test_preprocess[test_perm]
    test_data = np.double(test_data)
    test_data = test_data / 255.0
    test_label = test_label_preprocess[test_perm]

    # Feature selection
    # Your code here.
    """ many features which values are exactly the same for all data point
     So we can remove those value points since there is nothing to learn from them
    print('preprocess done')
    #print test_data
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for row in test_data:
        print row
        if any(row) == any(row):
            del row"""

    count = []
    x = len(train_data[1])
    i = 0
    while i != x:
        if np.std(train_data[:, i]) < .05:
            count.append(i)
        i += 1

    train_data = np.delete(train_data, count, axis=1)
    validation_data = np.delete(validation_data, count, axis=1)
    test_data = np.delete(test_data, count, axis=1)

    # print(len(test_data[0]))
    # print(len(validation_data[0]))
    # print(len(train_data[0]))



    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, t_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    '''
    n = len(training_data)
    training_data = np.ones((training_data.shape[0], 1))
    np.append(training_data, training_data, axis=1)
    a_j = np.dot(w1, np.transpose(training_data))
    z_j = sigmoid(a_j)
    bias = np.ones((z_j.shape[0], 1))
    z_j = np.append(z_j, bias, axis=1)
    b_l = np.dot(w2, np.transpose(z_j))

    o_l = sigmoid(b_l)
    print(o_l)
    obj_val = (-1 * 1 / n) * np.sum(
        np.sum((training_label.shape * np.log(o_l) + (np.subtract(1, training_label.shape)) * (np.log(1 - o_l)))))
    delta_l = np.subtract(o_l, training_label.shape)
    eq_9w2 = np.outer(delta_l, z_j)
    eq_12w1 = np.multiply(
        np.multiply(
            np.multiply(
                np.subtract(1, z_j), z_j), n_input), np.dot(np.transpose(delta_l), w2))

    sum1 = np.sum(np.square(w1))
    sum2 = np.sum(np.square(w2))

    j = obj_val + ((lambdaval / (2 * n)) * (sum1 + sum2))

    obj_val = j

    eq_9w2 = (np.sum(eq_9w2,  (lambdaval * w2))) / n
    eq_12w1 = (np.sum(eq_12w1, (lambdaval * w1))) / n'''

    # Your code here
    t_label = t_label.astype(int)
    #print(training_label)
    n = len(training_data)
    w1_g = np.zeros((n,n_hidden+1))
    w2_g = np.zeros((n,n_hidden+1))
    in_b = np.ones((len(training_data), 1))
    training_data = np.append(training_data, in_b, axis=1)
    h_out = np.transpose(sigmoid(np.dot(w1, np.transpose(training_data))))
    h_in_b = np.ones((h_out.shape[0],1))
    h_out = np.append(h_out, h_in_b, axis=1)
    i_out =  np.transpose(sigmoid(np.dot(w2, np.transpose(h_out))))
    yp = np.zeros((n,n_class))
    l_en = np.identity((n_class))
    t_en = np.identity(2)
    index = 0
    while index < (n):
        yp[index] = l_en[t_label[index]]
        index += 1
    y = np.multiply(yp, np.log(i_out)) + np.multiply((1 - yp) , np.log(1 - i_out))
    orange_juice = np.sum(-y) / n
    a_s = i_out-yp
    g2 = np.dot(np.transpose(a_s), h_out)
    asdotw2 = np.dot(a_s, w2)
    g1 = np.multiply(np.multiply(np.subtract(1, h_out), h_out), asdotw2)
    g1 = np.dot(np.transpose(g1), training_data)
    g1 = g1[:-1]
    #print(g1, g2)
    sum1 = np.sum(np.square(w1))
    sum2 = np.sum(np.square(w2))
    #print('orange juice', orange_juice)
    obj_val = orange_juice + np.multiply((lambdaval/np.multiply(2,n)), (sum1 + sum2))
    #print(obj_val)
    g1 = (g1 + np.multiply(lambdaval, w1))/n
    g2 = (g2 + np.multiply(lambdaval, w2))/n
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((g1.flatten(), g2.flatten()), 0)
    #print(obj_grad)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.
    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % labels: a column vector of predicted labels"""
    data_size = len(data)
    labels = np.ones(data_size)
    i = 0
    #print(data_size)
    while i < data_size:
        o= np.zeros(n_class)
        bias = np.array([1])
        t = np.concatenate((data[i], bias))

        escondido = np.dot(w1, t)
        index = 0
        while index < len(escondido):
            escondido[index] = sigmoid(escondido[index])
            index += 1

        secret = np.concatenate((escondido, bias))

        fuera = np.dot(w2, secret)
        top = 0
        test = 0
        index2 = 0
        while index2 < (n_class):
            o[index2] = sigmoid(fuera[index2])
            if o[index2] > top:
                top = o[index2]
                test = index2
            index2 += 1
        labels[i] = float(test)
        i += 1
    return labels

"""**************Neural Network Script Starts here********************************"""


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')