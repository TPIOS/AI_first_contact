import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *
from optimization import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# parameters, grads, learning_rate = update_parameters_with_gd_test_case()

# parameters = update_parameters_with_gd(parameters, grads, learning_rate)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))

# X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
# mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

# print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
# print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
# print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
# print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
# print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
# print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
# print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

# parameters = initialize_velocity_test_case()

# v = initialize_velocity(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))

# parameters, grads, v = update_parameters_with_momentum_test_case()

# parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))

# parameters = initialize_adam_test_case()

# v, s = initialize_adam(parameters)
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))

# parameters, grads, v, s = update_parameters_with_adam_test_case()
# parameters, v, s  = update_parameters_with_adam(parameters, grads, v, s, t = 2)

# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))
# print("v[\"dW1\"] = " + str(v["dW1"]))
# print("v[\"db1\"] = " + str(v["db1"]))
# print("v[\"dW2\"] = " + str(v["dW2"]))
# print("v[\"db2\"] = " + str(v["db2"]))
# print("s[\"dW1\"] = " + str(s["dW1"]))
# print("s[\"db1\"] = " + str(s["db1"]))
# print("s[\"dW2\"] = " + str(s["dW2"]))
# print("s[\"db2\"] = " + str(s["db2"]))

train_X, train_Y = load_dataset()
# # train 3-layer model
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# # Predict
# predictions = predict(train_X, train_Y, parameters)

# # Plot decision boundary
# plt.title("Model with Gradient Descent optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# # train 3-layer model
# layers_dims = [train_X.shape[0], 5, 2, 1]
# parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# # Predict
# predictions = predict(train_X, train_Y, parameters)

# # Plot decision boundary
# plt.title("Model with Momentum optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5,2.5])
# axes.set_ylim([-1,1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)