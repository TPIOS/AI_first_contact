from numpy import *
from PIL import Image
import sys, os
def pca(X):
    """主成分分析：
       输入：矩阵X，其中该矩阵中存储训练数据，每一行为一条训练数据
       返回：投影矩阵（按照维度的重要性排序），方差和均值"""
    
    num_data, dim = X.shape
    mean_X = X.mean(axis=0)
    X = X - mean_X

    if dim > num_data:
        M = dot(X, X.T)
        e, EVd = linalg.eigh(M)
        tmp = dot(X.T, EVd).T
        V = tmp[::-1]
        S = sqrt(e)[::-1]
        for i in range(V.shape[1]):
            V[:,i] /= S
    else:
        U, S, V = linalg.svd(X)
        V = V[:num_data]
    
    return V, S, mean_X