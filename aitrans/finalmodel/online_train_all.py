import LiveStreamingEnv.fixed_env as fixed_env
import LiveStreamingEnv.load_trace as load_trace
import time
import numpy as np
import ABRtrainAll as ABR # 注意这里，用的是ABRtrain
import pandas as pd
import random
from numpy.linalg import norm

# 梯度下降系数
alpha = 10e-21
alpha2 = 10e-15
videos = ['AsianCup_China_Uzbekistan', 'YYF_2018_08_12', 'Fengtimo_2018_11_3', 'game', 'room', 'sports']

def test(user_id):
    TRAIN_TRACES = './network_trace/'   #train trace path setting,
    video_size_file = './video_trace/{}/frame_trace_'.format(video)      #video trace path setting,
    LogFile_Path = "./log/"                #log file trace path setting,
    DEBUG = False
    # load the trace
    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TRAIN_TRACES)
    #random_seed
    random_seed = 2
    count = 0
    video_count = 0
    FPS = 25
    frame_time_len = 0.04
    reward_all_sum = 0
    #init 
    #setting one:
    #     1,all_cooked_time : timestamp
    #     2,all_cooked_bw   : throughput
    #     3,all_cooked_rtt  : rtt
    #     4,agent_id        : random_seed
    #     5,logfile_path    : logfile_path
    #     6,VIDEO_SIZE_FILE : Video Size File Path
    #     7,Debug Setting   : Debug
    net_env = fixed_env.Environment(all_cooked_time=all_cooked_time,
                                  all_cooked_bw=all_cooked_bw,
                                  random_seed=random_seed,
                                  logfile_path=LogFile_Path,
                                  VIDEO_SIZE_FILE=video_size_file,
                                  Debug = DEBUG)
    
    abr = ABR.Algorithm()
    abr_init = abr.Initial()

    BIT_RATE      = [500.0,850.0,1200.0,1850.0] # kpbs
    TARGET_BUFFER = [2.0,3.0]   # seconds
    # ABR setting
    RESEVOIR = 0.5
    CUSHION  = 2

    cnt = 0
    # defalut setting
    last_bit_rate = 0
    bit_rate = 0
    target_buffer = 0

    # QOE setting
    reward_frame = 0
    reward_all = 0
    SMOOTH_PENALTY= 0.02
    REBUF_PENALTY = 1.5
    LANTENCY_PENALTY = 0.005
    # past_info setting
    past_frame_num  = 7500
    S_time_interval = [0] * past_frame_num
    S_send_data_size = [0] * past_frame_num
    S_chunk_len = [0] * past_frame_num
    S_rebuf = [0] * past_frame_num
    S_buffer_size = [0] * past_frame_num
    S_end_delay = [0] * past_frame_num
    S_chunk_size = [0] * past_frame_num
    S_play_time_len = [0] * past_frame_num
    S_decision_flag = [0] * past_frame_num
    S_buffer_flag = [0] * past_frame_num
    S_cdn_flag = [0] * past_frame_num
    # params setting
   
    while True:
        reward_frame = 0
        # input the train steps
        #if cnt > 5000:
            #plt.ioff()
        #    break
        #actions bit_rate  target_buffer
        # every steps to call the environment
        # time           : physical time 
        # time_interval  : time duration in this step
        # send_data_size : download frame data size in this step
        # chunk_len      : frame time len
        # rebuf          : rebuf time in this step          
        # buffer_size    : current client buffer_size in this step          
        # rtt            : current buffer  in this step          
        # play_time_len  : played time len  in this step          
        # end_delay      : end to end latency which means the (upload end timestamp - play end timestamp)
        # decision_flag  : Only in decision_flag is True ,you can choose the new actions, other time can't Becasuse the Gop is consist by the I frame and P frame. Only in I frame you can skip your frame
        # buffer_flag    : If the True which means the video is rebuffing , client buffer is rebuffing, no play the video
        # cdn_flag       : If the True cdn has no frame to get 
        # end_of_video   : If the True ,which means the video is over.
        time,time_interval, send_data_size, chunk_len,\
               rebuf, buffer_size, play_time_len,end_delay,\
                cdn_newest_id, download_id, cdn_has_frame, decision_flag,\
                buffer_flag, cdn_flag, end_of_video = net_env.get_video_frame(bit_rate,target_buffer)

        # S_info is sequential order
        S_time_interval.pop(0)
        S_send_data_size.pop(0)
        S_chunk_len.pop(0)
        S_buffer_size.pop(0)
        S_rebuf.pop(0)
        S_end_delay.pop(0)
        S_play_time_len.pop(0)
        S_decision_flag.pop(0)
        S_buffer_flag.pop(0)
        S_cdn_flag.pop(0)

        S_time_interval.append(time_interval)
        S_send_data_size.append(send_data_size)
        S_chunk_len.append(chunk_len)
        S_buffer_size.append(buffer_size)
        S_rebuf.append(rebuf)
        S_end_delay.append(end_delay)
        S_play_time_len.append(play_time_len)
        S_decision_flag.append(decision_flag)
        S_buffer_flag.append(buffer_flag)
        S_cdn_flag.append(cdn_flag)        

        # QOE setting 
        if not cdn_flag:
            reward_frame = frame_time_len * float(BIT_RATE[bit_rate]) / 1000  - REBUF_PENALTY * rebuf - LANTENCY_PENALTY  * end_delay
        else:
            reward_frame = -(REBUF_PENALTY * rebuf)

        if decision_flag or end_of_video:
            # reward formate = play_time * BIT_RATE - 4.3 * rebuf - 1.2 * end_delay
            reward_frame += -1 * SMOOTH_PENALTY * (abs(BIT_RATE[bit_rate] - BIT_RATE[last_bit_rate]) / 1000)
            # last_bit_rate
            last_bit_rate = bit_rate

            # -------------------------------------------Your Althgrithom ------------------------------------------- 
            # which part is the althgrothm part ,the buffer based , 
            # if the buffer is enough ,choose the high quality
            # if the buffer is danger, choose the low  quality
            # if there is no rebuf ,choose the low target_buffer

        
            bit_rate , target_buffer = abr.run(time,S_time_interval,S_send_data_size,S_chunk_len,S_rebuf,S_buffer_size, S_play_time_len,S_end_delay,S_decision_flag,S_buffer_flag,S_cdn_flag, end_of_video, cdn_newest_id, download_id,cdn_has_frame,abr_init)

            # !! This is for online training
            if abr.max_predict != None:
                # print(current_sum, abr.max_predict)

                # 差值
                dif = float(abr.max_predict - reward_frame)
                # dif2 = (abr.max_predict_next - np.array([[S_time_interval[-1],S_send_data_size[-1],S_chunk_len[-1],S_rebuf[-1],S_buffer_size[-1], S_play_time_len[-1],S_end_delay[-1],S_decision_flag[-1],S_buffer_flag[-1],S_cdn_flag[-1], end_of_video]])).reshape(1, -1)
                # exit()
                if abs(dif)>100: # 如果差值爆炸了，马上停下来
                    print(dif, abr.last_choice)

                    input()
                # print(norm(dif2), abr.last_choice)
                
                # if current_sum>abr.max_predict:
                    # print(dif, abr.last_choice, current_sum)
                # exit()
                # print(abr.regs[abr.last_choice[0]][abr.last_choice[1]].coef_ )

                # reg是上一次决策所选的predict score模型
                reg = abr.regs[abr.last_choice[0]][abr.last_choice[1]]
                # reg.intercept_ -= 10e-9 * dif
                # 对coef系数进行梯度下降（intercept是常数）
                reg.coef_ -= alpha * dif * abr.last_x.reshape(13)
                reg.intercept_ -= alpha * dif

                # next_reg = abr.next_regs[abr.last_choice[0]][abr.last_choice[1]]
                # print(np.concatenate([abr.last_x_2]*11, axis=0))
                # exit()
                # next_reg.coef_ -= alpha2 * dif2 * np.concatenate([abr.last_x_2]*11, axis=0)
                # next_reg.intercept_ -= alpha2 * dif2.reshape(-1)
                
               


            # print("dataframe:\n\n\n", pd.DataFrame([[time,S_time_interval[-1],S_send_data_size[-1],S_chunk_len[-1],S_rebuf[-1],S_buffer_size[-1], S_play_time_len[-1],S_end_delay[-1],S_decision_flag[-1],S_buffer_flag[-1],S_cdn_flag[-1], end_of_video, 0, None]], columns = ["time", "S_time_interval", "S_send_data_size", "S_chunk_len", "S_rebuf", "S_buffer_size", "S_play_time_len","S_end_delay", "S_decision_flag", "S_buffer_flag","S_cdn_flag", "end_of_video", "decision", 'value']))
            #**
            # history = pd.concat([history, pd.DataFrame([[time,S_time_interval[-1],S_send_data_size[-1],S_chunk_len[-1],S_rebuf[-1],S_buffer_size[-1], S_play_time_len[-1],S_end_delay[-1],S_decision_flag[-1],S_buffer_flag[-1],S_cdn_flag[-1], end_of_video, decision, None]], columns = ["time", "S_time_interval", "S_send_data_size", "S_chunk_len", "S_rebuf", "S_buffer_size", "S_play_time_len","S_end_delay", "S_decision_flag", "S_buffer_flag","S_cdn_flag", "end_of_video", "decision", 'value'])])
            # print(history)
            # ------------------------------------------- End  ------------------------------------------- 
            
        if end_of_video:
            print("video count", video_count, reward_all)

            # 这个打开可以监测系数/常数值
            # print(' '.join([
                # str(abr.regs[i][j].intercept_) for i in range(4) for j in range(2)
            # ]))
            # input()
            reward_all_sum += reward_all / 1000
            video_count += 1
            if video_count >= len(all_file_names):
                    break
            cnt = 0
            last_bit_rate = 0
            reward_all = 0
            bit_rate = 0
            target_buffer = 0

            S_time_interval = [0] * past_frame_num
            S_send_data_size = [0] * past_frame_num
            S_chunk_len = [0] * past_frame_num
            S_rebuf = [0] * past_frame_num
            S_buffer_size = [0] * past_frame_num
            S_end_delay = [0] * past_frame_num
            S_chunk_size = [0] * past_frame_num
            S_play_time_len = [0] * past_frame_num
            S_decision_flag = [0] * past_frame_num
            S_buffer_flag = [0] * past_frame_num
            S_cdn_flag = [0] * past_frame_num
            
        
        reward_all += reward_frame
        
    # history.to_csv('randomData/t_{}.csv'.format(id))
    

    # !! 这是梯度下降之后保存一下模型
    abr.save_model()

    return reward_all_sum

# a = test('aaa')
# print(a)

# 这是循环一遍模式
for video in videos:
    print(video)
    a = test("aaa")
    print(a)

# 这是无限循环模式
# while True:
#     video = random.choice(videos)
#     print(video)
#     a = test("aaa")
#     print(a)  
