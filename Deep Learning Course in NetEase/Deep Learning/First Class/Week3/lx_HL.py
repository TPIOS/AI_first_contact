import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from function_need import *

np.random.seed(1)

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

# # Train the logistic regression classifier
# clf = sklearn.linear_model.LogisticRegressionCV();
# clf.fit(X.T, Y.T);

# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title("Logistic Regression")

# LR_predictions = clf.predict(X.T)

# parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

# # Plot the decision boundary
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))

# predictions = predict(parameters, X)
# print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

# Datasets
# noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

# datasets = {"noisy_circles": noisy_circles,
#             "noisy_moons": noisy_moons,
#             "blobs": blobs,
#             "gaussian_quantiles": gaussian_quantiles}

# ### START CODE HERE ### (choose your dataset)
# dataset = "noisy_moons"
# ### END CODE HERE ###

# X, Y = datasets[dataset]
# X, Y = X.T, Y.reshape(1, Y.shape[0])

# # make blobs binary
# if dataset == "blobs":
#     Y = Y%2

# # Visualize the data
# plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);