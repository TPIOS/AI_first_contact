import tensorflow as tf
import numpy as np
import pandas as pd
import csv

def load_dataset():
    data_path = "D:\\cbf\\University\\Year3 2018\\aitrans\\data\\"
    # for i in [0, 1]:
    #     for j in [0, 1, 2, 3]:
    for i in [0]:
        for j in [0]:
            filename = data_path+"merged_2_{}_{}.csv".format(str(i), str(j))
            df = pd.read_csv(filename)
            s_time_intervel = np.array(df[['S_time_interval']])
            s_send_data_size = np.array(df[['S_send_data_size']])
            s_frame_time_len = np.array(df[['S_frame_time_len']])
            s_frame_type = np.array(df[['S_frame_type']])
            s_buffer_size = np.array(df[['S_buffer_size']])
            s_end_delay = np.array(df[['S_end_delay']])
            rebuf_time = np.array(df[['rebuf_time']])
            buffer_flag = np.array(df[['buffer_flag']])
            # bitrate = np.array(df[['bitrate']])
            # buffer = np.array(df[['buffer']])
            score = np.array(df[['score']])
    
    return [s_time_intervel, s_send_data_size, s_frame_time_len, s_frame_type, s_buffer_size, s_end_delay, rebuf_time, buffer_flag, score]


data = load_dataset()
sess = tf.Session()
weights = []
bias=tf.Variable(tf.zeros([1]))

x1 = tf.placeholder(tf.float32)
w1 = tf.Variable(np.random.uniform(0.0, 1.0))
x2 = tf.placeholder(tf.float32)
w2 = tf.Variable(np.random.uniform(10000.0, 20000.0))
x3 = tf.placeholder(tf.float32)
w3 = tf.Variable(np.random.uniform(0.0, 1.0))
x4 = tf.placeholder(tf.float32)
w4 = tf.Variable(np.random.uniform(0.0, 1.0))
x5 = tf.placeholder(tf.float32)
w5 = tf.Variable(np.random.uniform(0.0, 1.0))
x6 = tf.placeholder(tf.float32)
w6 = tf.Variable(np.random.uniform(0.0, 1.0))
x7 = tf.placeholder(tf.float32)
w7 = tf.Variable(np.random.uniform(0.0, 1.0))
x8 = tf.placeholder(tf.float32)
w8 = tf.Variable(np.random.uniform(0.0, 1.0))
y_data = tf.placeholder(tf.float32)

y = x1*w1 + x2*w2 + x3*w3 + x4*w4 + x5*w5 + x6*w6 + x7*w7 + x8*w8 + bias

loss = tf.nn.l2_loss(y - y_data)

train=tf.train.AdagradOptimizer(2).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for i in range(35100):
    _, loss_val = sess.run([train, loss], 
        {x1:data[0][i*10:(i+1)*10],
         x2:data[1][i*10:(i+1)*10],
         x3:data[2][i*10:(i+1)*10],
         x4:data[3][i*10:(i+1)*10],
         x5:data[4][i*10:(i+1)*10],
         x6:data[5][i*10:(i+1)*10],
         x7:data[6][i*10:(i+1)*10],
         x8:data[7][i*10:(i+1)*10],
         y_data:data[-1][i*10:(i+1)*10]})
    
    if i % 1000 == 0:
        print("{}, {}, {}, {}".format(i, loss_val, sess.run([w1]), sess.run([w2])))
