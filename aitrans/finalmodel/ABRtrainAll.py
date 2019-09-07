## 主要区别：1. 把bitrate和buffer的改变作为一个特征加以训练（而非直接定义惩罚值）；2. 保存模型
# import tensorflow as tf
# import xgboost as xgb
import numpy as np
# import sklearn.svm
import random

from sklearn.externals import joblib
import time as ptime
# import joblib


def max_next_value(bitrate, target_buffer, regs, next_regs, current_state, depth):
    if depth==0:
        return 0
    
    # initData = np.array(
    #     # [
    #     # 0.012890761852506274,
    #     # 18222.911351223454,
    #     # 0.03999999999772906,
    #     # 0.0020015655001585963,
    #     # n1,
    #     # 0.011778484918187428,
    #     # n2,
    #     # 0.9992887704601725,
    #     # 0.038526811241088504,
    #     # 0.0,
    #     # 0.
    #     # ]
    #     [
    #         0.014156946990699481, 17771.584478760316, 0.039999999997843406, 0.005216416243339166, n1, 0.01141073540361565, n2, 0.9992325425132007, 0.07149403818327145, 0.0, 0.0007674574867992609, 0, 0
    #     ]
    # ).reshape(1, -1)

    
    x_2 = current_state
    current_state = np.concatenate([x_2, [[0,0]]], axis=1).reshape(1, -1)
    max_v = -10000
    for i in [0, 2]:
        for j in range(2):
            next_state = next_regs[i][j].predict(x_2)[0]

            # v = regs[i][j].predict(initData) + max_next_value(i, j, regs, next_regs, n1, n2, depth-1) * 0.99

            # 这11和12分别是bitrate和buffer的改变
            current_state[0, 11] = abs(bitrate-i)
            current_state[0, 12] = abs(target_buffer-j)
            v = regs[i][j].predict(current_state) + max_next_value(i, j, regs, next_regs, next_state, depth-1) * 0.99
            # if bitrate!=i:
            # v -= 0.01 * abs(bitrate-i)
            # if target_buffer!=j:
            # v -= 0.5 * abs(target_buffer-j)
            max_v = max(v, max_v)
    # print(max_v)
    return max_v

class Algorithm:
    def __init__(self):
    # fill your init vars
        self.buffer_size = 0
        # self.param = {'max_depth': 30, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 4}
        # self.svm =  joblib.load("train_model3.m")
        # self.svm =  joblib.load("train_model4.m")
        # self.svm =  joblib.load("/home/team/WwW/submit/results/train_model3.m")
        # self.bst = xgb.Booster(self.param)
        # self.bst = xgb.Booster(self.param)
        # self.bst.load_model('/home/team/WwW/submit/results/bstModel.model')
        # self.bst.load_model('bstModel.model')
        # self.bst.load_model('bst_1.model')
        

        # change代表是多了change of bitrate 和 buffer的模型！！！
        self.regs = [
            # joblib.load("reg_{}.joblib".format(i)) for i in range(4)
            # [joblib.load("reg_{}_{}_change.joblib".format(i, j)) for j in range(2)] for i in range(4)
            [joblib.load("reg_{}_{}_change_all.joblib".format(i, j)) for j in range(2)] for i in range(4)
            # [joblib.load("reg_{}_{}_double.joblib".format(i, j)) for j in range(2)] for i in range(4)
        ]
        self.next_regs = [
            [joblib.load("next_predict_{}_{}.joblib".format(i, j)) for j in range(2)] for i in range(4)
            # [joblib.load("next_{}_{}_newnet.joblib".format(i, j)) for j in range(2)] for i in range(4)
        ]

        self.last_choice = (-1, -1)
        self.max_predict = None
        self.last_x = None
        self.time = []

        
    # Intial 
    def Initial(self):
        IntialVars = []
            
        return IntialVars

    #Define your algorithm
    def run(self, time, S_time_interval, S_send_data_size, S_chunk_len, S_rebuf, S_buffer_size, S_play_time_len,S_end_delay, S_decision_flag, S_buffer_flag,S_cdn_flag, end_of_video, cdn_newest_id, download_id,cdn_has_frame, IntialVars):
        # t1 = ptime.clock()

        # 这里也是多了两位
        x = np.array([[S_time_interval[-1],S_send_data_size[-1],S_chunk_len[-1],S_rebuf[-1],S_buffer_size[-1], S_play_time_len[-1],S_end_delay[-1],S_decision_flag[-1],S_buffer_flag[-1],S_cdn_flag[-1], end_of_video, 0, 0]])

        # x_2是去掉后两位（因为预测next state的模型没这两位）
        x_2 = x[:, :11]
        self.last_x = x
        self.last_x_2 = x_2

        max_r = max_predict =  -10000
        max_c = (0, 0)
        for i in [0, 2]:
            for j in range(2):
                
                next_state = self.next_regs[i][j].predict(x_2)
                
                # 后两位是根据bitrate和buffer的改变决定的
                x[0, 11] = abs(self.last_choice[0] - i)
                x[0, 12] = abs(self.last_choice[1] - j)

                # predict = self.regs[i][j].predict(x.reshape(1, -1)) - 0.01 * abs(self.last_choice[0] - i) - 0.5 * abs(self.last_choice[1]-j)
                predict = self.regs[i][j].predict(x.reshape(1, -1))
                # predict = self.regs[i][j].predict(np.concatenate([x, np.power(x[:, :7], 2)], axis=1))

                p = predict + max_next_value(i, j, self.regs, self.next_regs, next_state, 1) * 0.99
                # if self.last_choice[0]!=i:
                # p -= 0.01 * abs(self.last_choice[0] - i)
                # if self.last_choice[1]!=j:
                # p -= 0.5 * abs(self.last_choice[1]-j)
        #         print(p, i)
                if p>max_r:
                    max_r = p
                    max_c = (i, j)
                    max_predict = predict
                    max_predict_next = next_state
        
        self.last_choice = max_c
        self.max_predict = max_predict
        self.max_predict_next = max_predict_next

        # print(max_predict, max_c)
        bit_rate, target_buffer = max_c
        # self.time.append(ptime.clock() - t1)
        # bit_rate = int(self.bst.predict(x)[0])
        
        # bit_rate = 1
        # print(bit_rate)
        # RESEVOIR = 0.3
        # CUSHION =  0.8
        # bit_rate = 0
        # if S_buffer_size[-1] < RESEVOIR:
        #     bit_rate = 0
        # elif S_buffer_size[-1] >= CUSHION + CUSHION:
        #     bit_rate = 3
        # elif S_buffer_size[-1] >= RESEVOIR + CUSHION:
        #     bit_rate = 2
        
        # else:
        #     bit_rate = 1
        # target_buffer = 0
        # bit_rate = 1
        return bit_rate, target_buffer

        # If you choose other
        #......


    # Remember to remove!
    # 这是保存模型
    def save_model(self):
        for i in range(4):
            for j in range(2):
                joblib.dump(self.regs[i][j], "reg_{}_{}_change_all.joblib".format(i, j))
                # joblib.dump(self.next_regs[i][j], "next_{}_{}_all.joblib".format(i, j))