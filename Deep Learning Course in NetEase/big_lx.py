import math
import numpy as np
def sigmoid(x):
    """
    Compute sigmoid of x.
    Arguments:
    x -- A scalar or numpy array of any size
    Return:
    s -- sigmoid(x)
    """
    s = 1/(1+np.exp(-x))
    return s

def sigmoid_derivatives(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    Arguments:
    x -- A scalar or numpy array
    Return:
    ds -- Your computed gradient.
    """
    s = sigmoid(x)
    ds = s*(1-s)
    return ds

def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    v = image.reshape(image.shape[0]*image.shape[1]*image.shape[2], 1)
    return v

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    Argument:
    x -- A numpy matrix of shape (n, m)
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

def softmax(x):
    """
    Calculates the softmax for each row of the input x.
    Your code should work for a row vector and also for matrices of shape (n, m).
    Argument:
    x -- A numpy matrix of shape (n,m)
    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    s = x_exp / x_sum
    return s

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(abs(yhat-y))
    return loss

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    loss = np.sum((yhat-y)**2)
    return loss


yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))