import numpy as np
import matplotlib.pyplot as plt
from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *
from regularization import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()

# parameters = model(train_X, train_Y)
# print ("On the training set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model without regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

# parameters = model(train_X, train_Y, lambd = 0.7)
# print ("On the train set:")
# predictions_train = predict(train_X, train_Y, parameters)
# print ("On the test set:")
# predictions_test = predict(test_X, test_Y, parameters)

# plt.title("Model with L2-regularization")
# axes = plt.gca()
# axes.set_xlim([-0.75,0.40])
# axes.set_ylim([-0.75,0.65])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)