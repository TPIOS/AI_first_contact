import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from tensorflow_tutorial import *

# y_hat = tf.constant(36, name="y_hat")
# y = tf.constant(39, name="y")

# loss = tf.Variable((y - y_hat)**2, name="loss")

# init = tf.global_variables_initializer()

# with tf.Session() as session:
#     session.run(init)
#     print(session.run(loss))

# print( "result = " + str(linear_function()))

# print ("sigmoid(0) = " + str(sigmoid(0)))
# print ("sigmoid(12) = " + str(sigmoid(12)))

# logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
# cost = cost(logits, np.array([0,0,1,1]))
# print ("cost = " + str(cost))

# labels = np.array([1,2,3,0,2,1])
# one_hot = one_hot_matrix(labels, C = 4)
# print ("one_hot = " + str(one_hot))
# print ("ones = " + str(ones([3])))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
# index = 0
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

# plt.show()
# Flatten the training and test images

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

# print ("number of training examples = " + str(X_train.shape[1]))
# print ("number of test examples = " + str(X_test.shape[1]))
# print ("X_train shape: " + str(X_train.shape))
# print ("Y_train shape: " + str(Y_train.shape))
# print ("X_test shape: " + str(X_test.shape))
# print ("Y_test shape: " + str(Y_test.shape))
# X, Y = create_placeholders(12288, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))
# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))

# tf.reset_default_graph()

# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     print("Z3 = " + str(Z3))

# tf.reset_default_graph()

# with tf.Session() as sess:
#     X, Y = create_placeholders(12288, 6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y)
#     print("cost = " + str(cost))

parameters = model(X_train, Y_train, X_test, Y_test)