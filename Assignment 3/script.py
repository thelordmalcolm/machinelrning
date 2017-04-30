import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize
from sklearn.svm import SVC
import pickle


def preprocess():
    """
     Input:
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
    """

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    n_feature = mat.get("train1").shape[1]
    n_sample = 0
    for i in range(10):
        n_sample = n_sample + mat.get("train" + str(i)).shape[0]
    n_validation = 1000
    n_train = n_sample - 10 * n_validation

    # Construct validation data
    validation_data = np.zeros((10 * n_validation, n_feature))
    for i in range(10):
        validation_data[i * n_validation:(i + 1) * n_validation, :] = mat.get("train" + str(i))[0:n_validation, :]

    # Construct validation label
    validation_label = np.ones((10 * n_validation, 1))
    for i in range(10):
        validation_label[i * n_validation:(i + 1) * n_validation, :] = i * np.ones((n_validation, 1))

    # Construct training data and label
    train_data = np.zeros((n_train, n_feature))
    train_label = np.zeros((n_train, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("train" + str(i)).shape[0]
        train_data[temp:temp + size_i - n_validation, :] = mat.get("train" + str(i))[n_validation:size_i, :]
        train_label[temp:temp + size_i - n_validation, :] = i * np.ones((size_i - n_validation, 1))
        temp = temp + size_i - n_validation

    # Construct test data and label
    n_test = 0
    for i in range(10):
        n_test = n_test + mat.get("test" + str(i)).shape[0]
    test_data = np.zeros((n_test, n_feature))
    test_label = np.zeros((n_test, 1))
    temp = 0
    for i in range(10):
        size_i = mat.get("test" + str(i)).shape[0]
        test_data[temp:temp + size_i, :] = mat.get("test" + str(i))
        test_label[temp:temp + size_i, :] = i * np.ones((size_i, 1))
        temp = temp + size_i

    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis=0)
    index = np.array([])
    for i in range(n_feature):
        if (sigma[i] > 0.001):
            index = np.append(index, [i])
    train_data = train_data[:, index.astype(int)]
    validation_data = validation_data[:, index.astype(int)]
    test_data = test_data[:, index.astype(int)]

    # Scale data to 0 and 1
    train_data /= 255.0
    validation_data /= 255.0
    test_data /= 255.0

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def blrObjFunction(initialWeights, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector (w_k) of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector (y_k) of size N x 1 where each entry can be either 0 or 1 representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    train_data, labeli = args

    n_data = train_data.shape[0]
    n_features = train_data.shape[1]

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    bias = np.ones((n_data, 1))
    data_bias = np.concatenate((train_data, bias), axis=1)
    weight = np.reshape(initialWeights, (n_features + 1, 1))
    db_dot_w = np.dot(data_bias, weight)
    theta_n = sigmoid(db_dot_w)
    try:
        lg = np.log(theta_n)
    except:
        lg = 0
    theta = np.array(bias) - np.array(theta_n)
    lg_t = np.log(theta)
    lbl_dot_lg = np.dot(np.transpose(labeli), lg)
    bias_diff_lbl = np.array(bias) - np.array(labeli)
    bias_df_dot_lgt = np.dot(bias_diff_lbl.transpose(), lg_t)
    e = np.add(lbl_dot_lg, bias_df_dot_lgt)
    error = (-1*np.sum(e)) / n_data

    theta_diff_lbl = np.subtract(theta_n, labeli)
    datab_dot_thetadiff = np.dot(data_bias.transpose(), theta_diff_lbl)

    error_grad = datab_dot_thetadiff / n_data

    return error, error_grad.flatten()


def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data
    v = np.ones((data.shape[0], 1))
    concat_v_and_data = np.concatenate((data, v), axis=1)
    dot_concat_and_w = np.dot(concat_v_and_data, W)
    theta = sigmoid(dot_concat_and_w)
    j = 0
    while j < label.shape[0]:
        temp = np.amax(theta[i])
        label[i] = temp
        j += 1
    return label


def mlrObjFunction(params, *args):
    """
    mlrObjFunction computes multi-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output:
        error: the scalar value of error function of multi-class logistic regression
        error_grad: the vector of size (D+1) x 10 representing the gradient of
                    error function
    """
    n_data = train_data.shape[0]
    n_feature = train_data.shape[1]
    error = 0
    error_grad = np.zeros((n_feature + 1, n_class))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return error, error_grad


def mlrPredict(W, data):
    """
     mlrObjFunction predicts the label of data given the data and parameter W
     of Logistic Regression

     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D

     Output:
         label: vector of size N x 1 representing the predicted label of
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0], 1))

    ##################
    # YOUR CODE HERE #
    ##################
    # HINT: Do not forget to add the bias term to your input data

    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# number of classes
n_class = 10

# number of training samples
n_train = train_data.shape[0]

# number of features
n_feature = train_data.shape[1]

Y = np.zeros((n_train, n_class))
for i in range(n_class):
    Y[:, i] = (train_label == i).astype(int).ravel()

# Logistic Regression with Gradient Descent
W = np.zeros((n_feature + 1, n_class))
initialWeights = np.zeros((n_feature + 1, 1))
opts = {'maxiter': 100}
for i in range(n_class):
    labeli = Y[:, i].reshape(n_train, 1)
    args = (train_data, labeli)
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
    W[:, i] = nn_params.x.reshape((n_feature + 1,))

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

"""
Script for Support Vector Machine
"""

f1 = open('params.pickle', 'wb')
pickle.dump(W, f1)
f1.close()


print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

'''linear kernal'''
agh = SVC(kernel='linear');
agh.fit(train_data, train_label.flatten())
accuracy = agh.score(train_data, train_label.flatten());
print('Accuracy for linear train data: {}%'.format(accuracy*100))
accuracy = agh.score(validation_data, validation_label.flatten());
print('Accuracy for the linear validation data: {}%'.format(accuracy*100))
accuracy = agh.score(test_data,test_label.flatten());
print('Accuracy for the linear test data: {}%'.format(accuracy*100))

'''Kernel RBF Gamma = 1.'''
agh = SVC(kernel='rbf', gamma=1.);
agh.fit(train_data, train_label.flatten())
accuracy = agh.score(train_data, train_label.flatten());
print('Accuracy for rbf, gamma = 1.0 train data: {}%'.format(accuracy*100))
accuracy = agh.score(validation_data, validation_label.flatten());
print('Accuracy for rbf, gamma = 1.0 validation data: {}%'.format(accuracy*100))
accuracy = agh.score(test_data,test_label.flatten());
print('Accuracy for rbf, gamma = 1.0 test data: {}%'.format(accuracy*100))

'''Kernel RBF default'''
agh = SVC(kernel='rbf', gamma=1.);
agh.fit(train_data, train_label.flatten())
accuracy = agh.score(train_data, train_label.flatten());
print('Accuracy for RBF default train data: {}%'.format(accuracy*100))
accuracy = agh.score(validation_data, validation_label.flatten());
print('Accuracy for RBF default validation data: {}%'.format(accuracy*100))
accuracy = agh.score(test_data,test_label.flatten());
print('Accuracy for RBF default test data: {}%'.format(accuracy*100))

'''Kernel RBF Varying C'''
agh = SVC(c=1,kernel='rbf');
agh.fit(train_data,train_label.flatten());
accuracy = agh.score(train_data,train_label.flatten());
print('Accuracy for RBF C = {} train data: {}%'.format(1, accuracy*100))
accuracy = agh.score(validation_data, validation_label.flatten());
print('Accuracy for RBF C = {} validation data: {}%'.format(1, accuracy*100))
accuracy = agh.score(test_data,test_label.flatten());
print('Accuracy for RBF C = {} test data: {}%'.format(1, accuracy*100))

for i in range(10,110,10):
    agh = SVC(c=i, kernel='rbf');
    agh.fit(train_data, train_label.flatten());
    accuracy = agh.score(train_data, train_label.flatten());
    print('Accuracy for RBF C = {} train data: {}%'.format(i, accuracy * 100))
    accuracy = agh.score(validation_data, validation_label.flatten());
    print('Accuracy for RBF C = {} validation data: {}%'.format(i, accuracy * 100))
    accuracy = agh.score(test_data, test_label.flatten());
    print('Accuracy for RBF C = {} test data: {}%'.format(i, accuracy * 100))

"""
Script for Extra Credit Part
"""
# FOR EXTRA CREDIT ONLY
W_b = np.zeros((n_feature + 1, n_class))
initialWeights_b = np.zeros((n_feature + 1, n_class))
opts_b = {'maxiter': 100}

args_b = (train_data, Y)
nn_params = minimize(mlrObjFunction, initialWeights_b, jac=True, args=args_b, method='CG', options=opts_b)
W_b = nn_params.x.reshape((n_feature + 1, n_class))

# Find the accuracy on Training Dataset
predicted_label_b = mlrPredict(W_b, train_data)
print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label_b == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label_b = mlrPredict(W_b, validation_data)
print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label_b == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label_b = mlrPredict(W_b, test_data)
print('\n Testing set Accuracy:' + str(100 * np.mean((predicted_label_b == test_label).astype(float))) + '%')
