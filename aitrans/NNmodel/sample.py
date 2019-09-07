import tensorflow as tf
import numpy as np
import matplotlib.pyplot  as plt

def data_show(x,y,w,b):
    plt.figure()
    plt.scatter(x,y,marker='.')
    plt.scatter(x,(w*x+b),marker='.')
    plt.show()

x_data=np.random.rand(100).astype(np.float32)
y_data=0.1*x_data + 0.3

Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))#平均分布的随机数
biases=tf.Variable(tf.zeros([1]))
y=Weights*x_data+biases
loss=tf.reduce_mean(tf.square(y-y_data)) #损失函数，reduce_mean:计算一个张量的各维度的元素的均值
optimizer=tf.train.GradientDescentOptimizer(0.5)#优化器 学习率选择#.GradientDescentOptimizer()实现梯度下降算法的优化器。
train=optimizer.minimize(loss)#优化器优化目标选择,使loss 最小
init=tf.global_variables_initializer() #初始化全变量节点

###训练部分
with tf.Session() as sess:
    sess.run(init)
    for i in range(200):
        sess.run(train)
        if i %20==0:
            print(i,sess.run(Weights),sess.run(biases))
            data_show(x_data,y_data,sess.run(Weights),sess.run(biases))