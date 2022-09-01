# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for trajectory generation Class'
__author__ = 'Li Peiran'

import os
import time
from collections import ChainMap
import DataLoaderLSTM as DataLoaderLSTM
import minitools
import jismesh.utils as ju
import numpy as np
import pandas as pd
import torch
import torch.autograd
import FNNGAN, DCGAN, GCN, WordLSTM
from pandas import DataFrame
from torch.autograd import Variable
from tqdm import tqdm

# 4.Define the global variables. (if exists)

#Define divice
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_img =1

class TrajGenerator():
    '''
    This is for traj_generator.
    '''
    def __call__(self):
        return 1

    def __init__(self, MODEL_TYPE,LOC_MODE,DATA_STRUCT_MODE,TRAIN_USER_NUM,HOLIDAY_MODE,ORI_LP_FOLDER,OD_LIB_PATH,temperature):
        '''
        This is for initial class.
        '''

        MODEL_FOLDER_PATH = '../assets/' + MODEL_TYPE + '_' + LOC_MODE + '/weekdaymode_' + \
                    str(1 - HOLIDAY_MODE) + '_user_' + str(TRAIN_USER_NUM) + '/' + DATA_STRUCT_MODE + '/'

        # Load Input Data: the labelled traj, input vector form
        if MODEL_TYPE == 'FNN':

            # 1. Load the input location data, vector form data, grah form data
            total_lp_code_list = [None]  # 如果有，给loc info file path
            matrix_form_df = None
            VECTOR_FORM_DF_PATH = MODEL_FOLDER_PATH + 'vector_form_df.csv'
            vector_form_df = pd.read_csv(VECTOR_FORM_DF_PATH, index_col=0).T
            vector_size = len(vector_form_df.columns)
            if LOC_MODE == 'Pure Gaussian Noise':
                loc_info_list = [0 for x in total_lp_code_list]
                z_dimension = 100
            elif LOC_MODE == 'With Location Info':
                loc_info_list = [x[-2:, :] for x in total_lp_code_list]
                z_dimension = loc_info_list[0].shape[1] * 2

            # Load Model Path
            if LOC_MODE == 'Pure Gaussian Noise':
                G_PATH = MODEL_FOLDER_PATH + '/_generator_epoch2.pth'
            elif LOC_MODE == 'With Location Info':
                G_PATH = MODEL_FOLDER_PATH + '/_generator_epoch2.pth'
            # D = GCN.graphDiscriminator(1,graph).to(device)
            G = FNNGAN.Generator(z_dimension, vector_size).to(device)
            G.load_state_dict(torch.load(G_PATH, map_location=device))

        elif MODEL_TYPE == 'DC':
            z_dimension = 100
            G = DCGAN.Generator(z_dimension).to(device)
            # G.load_state_dict(torch.load(G_PATH))

        elif MODEL_TYPE == 'GCN':

            # 1. Load the input location data, vector form data, grah form data
            total_lp_code_list = [None]  # 如果有，给loc info file path
            matrix_form_df = None
            VECTOR_FORM_DF_PATH = MODEL_FOLDER_PATH + 'vector_form_df.csv'
            GRAPH_FORM_PKL_PATH = MODEL_FOLDER_PATH + 'graph_form.pkl'
            vector_form_df = pd.read_csv(VECTOR_FORM_DF_PATH, index_col=0).T
            vector_size = len(vector_form_df.columns)
            graph = minitools.loadPKL(GRAPH_FORM_PKL_PATH)
            if LOC_MODE == 'Pure Gaussian Noise':
                loc_info_list = [0 for x in total_lp_code_list]
                z_dimension = 100
            elif LOC_MODE == 'With Location Info':
                loc_info_list = [x[-2:, :] for x in total_lp_code_list]
                z_dimension = loc_info_list[0].shape[1] * 2

            # Load Model Path
            if LOC_MODE == 'Pure Gaussian Noise':
                G_PATH = MODEL_FOLDER_PATH + '/_generator_epoch0.pth'
            elif LOC_MODE == 'With Location Info':
                G_PATH = MODEL_FOLDER_PATH + '/_generator_epoch0.pth'
            # D = GCN.graphDiscriminator(1,graph).to(device)
            G = GCN.graphGenerator_1(z_dimension, graph).to(device)
            G.load_state_dict(torch.load(G_PATH))

        if MODEL_TYPE == 'LSTM':
            sequence_length = 1
            G_PATH = '../assets/LSTM_None/user_40/None/lstm.pth'

            lstm_dataset = DataLoaderLSTM.Dataset(sequence_length)
            G = WordLSTM.WordLSTM(len(lstm_dataset.uniq_words)).to(device)
            G.load_state_dict(torch.load(G_PATH))

        # Load the labelled traj files

        lp_file_list = []
        minitools.getFilePath(ORI_LP_FOLDER, lp_file_list, dir_list=[], target_ext='.csv')

        # Load the traj library

        temp_folder_list = []
        od_lib_df_list = []
        od_lib_dict_file_list = []
        minitools.getFilePath(OD_LIB_PATH, od_lib_dict_file_list, temp_folder_list, '.pkl')
        OB_LIB_USER_NUM = len(od_lib_dict_file_list)
        # # Merge all lib (太慢了，暂时先不merge，用低配版的，逐人建lib，配不上也不搜寻其他人)
        od_lib_dict_list = []
        for i in range(len(od_lib_dict_file_list)):
            temp_od_lib_dict = minitools.loadPKL(od_lib_dict_file_list[i])
            od_lib_dict_list.extend(temp_od_lib_dict)
        total_od_lib_dict = dict(ChainMap(*od_lib_dict_list))
        minitools.savePKL(total_od_lib_dict,'../assets/total_od_lib_dict.pkl')

        # 5.Define the class (if exsists)
        # 6.Define the function (if exsists)

        #Initialize the class paramters
        self.lp_file_list = lp_file_list
        self.z_dimension = z_dimension
        self.od_lib_dict_file_list = od_lib_dict_file_list
        self.G = G
        self.vector_form_df = vector_form_df
        self.matrix_form_df = matrix_form_df
        self.loc_info_list = loc_info_list
        self.lstm_dataset = None
        self.OB_LIB_USER_NUM = OB_LIB_USER_NUM
        self.temperature = temperature
        self.DATA_STRUCT_MODE = DATA_STRUCT_MODE

        # return self.lp_file_list, self.z_dimension, self.od_lib_dict_file_list, \
        #        self.G, self.vector_form_df, self.matrix_form_df, self.loc_info_list, \
        #        self.lstm_dataset, self.OB_LIB_USER_NUM


    def vector2Matrix(self,x):
        if self.DATA_STRUCT_MODE =='matrix':
            out = x.view(-1, 1, self.matrix_form_df.shape[0], self.matrix_form_df.shape[1])
            return out
        elif self.DATA_STRUCT_MODE =='vector':
            out = x
            return out
        return 0

    def createRandomNoise(self,mode,add_info):
        z = 0
        if mode=='With Location Info':
            z_loc = Variable(torch.tensor(add_info, dtype=torch.float32)).cuda().reshape(1, -1)
            z_random = Variable(torch.randn(num_img, len(z_loc[0]))).cuda()  # 得到随机噪声
            z = z_random + z_loc
        elif mode=='Pure Gaussian Noise':
            z = Variable(torch.randn(num_img, self.z_dimension)).to(device) # for FNN
        else:
            print('Please Give Correct Mode Name.')
        return z

def wordSample(preds, temperature):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)  # 重新加权调整
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)  # 返回概率最大的字符下标

# 根据原始label了significant places的轨迹文件提取出significant place对应的经纬度
def lpFile2LocDict(current_lp_df):
    home_place_num = len(current_lp_df['home_label_order'].unique())-1
    work_place_num = len(current_lp_df['work_label_order'].unique())-1
    other_place_num = len(current_lp_df['other_label_order'].unique())-1
    home_places = []
    work_places = []
    other_places= []
    for k in range(home_place_num):
        home_places.append(current_lp_df[current_lp_df['home_label_order']==k][['home_lat','home_lon']].iloc[0].values)
    for k in range(work_place_num):
        work_places.append(current_lp_df[current_lp_df['work_label_order']==k][['work_lat','work_lon']].iloc[0].values)
    for k in range(other_place_num):
        other_places.append(current_lp_df[current_lp_df['other_label_order']==k][['other_lat','other_lon']].iloc[0].values)
    # 2.建立统一结构的lp_code-location字典
    home_location_dict = {'H_'+str(m):home_places[m] for m in range(len(home_places))}
    work_location_dict = {'W_'+str(m):work_places[m] for m in range(len(work_places))}
    other_location_dict = {'O_'+str(m):other_places[m] for m in range(len(other_places))}
    lp_location_dict = {}
    for d in [home_location_dict, work_location_dict, other_location_dict]:
        lp_location_dict.update(d)
    return lp_location_dict


# 根据生成的life pattern代码 + 某user的significant places + Traj library 生成一个人的一天轨迹
def lpPob2LpCode(temp_lp,DATA_STRUCT_MODE,vector_form_df,temperature):

    lp_code_list = []

    if DATA_STRUCT_MODE == 'vector':
        vector_form_df = vector_form_df.T
        vector_form_df.columns = ['hour','pob']
        vector_form_df['pob'] = temp_lp.values[0,:] # total_lp_code_list
        #add OD attributes
        vector_form_df['o'] = [x.split('.')[0] for x in vector_form_df.index.values]
        vector_form_df['d'] = [x.split('.')[1] for x in vector_form_df.index.values]
        for hour,lp_code in vector_form_df.groupby('hour'):

            if len(lp_code_list) != 0:
                last_choose_lp_code = lp_code_list[-1]
                last_choose_d = last_choose_lp_code.split('.')[-1]
                # filter o = last d
                lp_code = lp_code[lp_code['o']==last_choose_d]
                p = lp_code['pob'].values
            else:
                p = lp_code['pob'].values
                # p = MiniTools.normSum(p)
            choose_lp_code = lp_code.index[wordSample(p, temperature=temperature)]
            # if (sum(p) != 1):
            #     choose_lp_code = lp_code.index[np.argmax(p)]
            #     #continue
            # else:
            #     choose_lp_code = np.random.choice(lp_code.index, p=p.ravel())

            #save 一下 choose_lp_code
            lp_code_list.append(choose_lp_code)

    elif DATA_STRUCT_MODE == 'matrix':

        # 全部可能的情况有
        lp_code = temp_lp.columns
        for i in range(24):
            hour = i
            # 按照归一化之后的概率取值
            p = temp_lp.iloc[i].values
            #p = total_lp_code_list[uid][0:24, :][i]
            p = minitools.normSum(p)
            #DataFrame(p,index=lp_code).sort_values(by=0,ascending=False)
            #print(p)
            if sum(p)>12:
                choose_lp_code = lp_code[wordSample(p, temperature=temperature)]
            else:
                choose_lp_code = lp_code[np.argmax(p)]
            # if (sum(p) != 1):
            #     choose_lp_code = lp_code_type[np.argmax(p)]
            #     # continue
            # else:
            #     choose_lp_code = np.random.choice(lp_code_type, p=p.ravel())

            # save 一下 choose_lp_code
            lp_code_list.append(choose_lp_code)

    elif DATA_STRUCT_MODE == 'vector_lstm':
        lp_location_list  = []
        lp_location_list.append((temp_lp[0].split('.')[0],temp_lp[0].split('.')[1]))
        for i in range(1,len(temp_lp)-1):
            hour = temp_lp[i].split('.')[0]
            #print(temp_lp[i])
            snf_p = temp_lp[i].split('.')[1]
            if (hour > lp_location_list[-1][0]):
                lp_location_list.append((hour,snf_p))
        lp_code_list = lp_location_list

    if DATA_STRUCT_MODE != 'vector_lstm':
        lp_location_list = []
        lp_location_list.append((0, lp_code_list[0].split('.')[0])) # 1. considering state before -> filter ->sample
        for i in range(1,len(lp_code_list)*2):
            hour = int(i/2)
            temp = (hour, lp_code_list[hour].split('.')[i%2])
            if temp[1] != lp_location_list[-1][1]:
                lp_location_list.append(temp)
        lp_code_list = lp_location_list

    return lp_code_list

def lpCode2Traj(uid, lp_code_list,lp_location_dict,od_lib_dict):

    detail_traj_list = []
    day = 0
    for i in range(len(lp_code_list)-1):

        # 1. from sig places to OD pair
        o_time = int(lp_code_list[i][0])
        d_time = int(lp_code_list[i+1][0])
        o_label = lp_code_list[i][1]
        d_label = lp_code_list[i + 1][1]
        o_order = int(o_label.split('_')[1])
        d_order = int(d_label.split('_')[1])
        if o_time > d_time:
            day = day+1
        try:
            temp_o_location = lp_location_dict[o_label]
            temp_d_location = lp_location_dict[d_label]
            o_mesh = ju.to_meshcode(temp_o_location[0], temp_o_location[1], 4)
            d_mesh = ju.to_meshcode(temp_d_location[0], temp_d_location[1], 4)
        except:
            #print('Significant place not found,pass.')#%d or %d'%o_order%d_order)
            continue
        #2. 写入轨迹:a.如果不动，直接写入O的经纬度 b.如果动，通过dict插入轨迹，写入
        if o_label==d_label:
            MOVE_STATE = False
            detail_traj_list.append(
                {'day':day,'uid': uid, 'time': o_time, 'lat': temp_o_location[0], 'lon': temp_o_location[1]})
        else:
            MOVE_STATE = True
            try:
                temp_detail_traj_list = od_lib_dict[(o_mesh, d_mesh)] #从detial轨迹库中读取轨迹
            except:
                #print('Traj not found in the lib,pass.')
                continue
            seg_period = (temp_detail_traj_list[-1][0] - temp_detail_traj_list[0][0])/3600  # 计算这段轨迹的总耗时
            #a.如果总耗时大于OD本身的时间间隔，则跳过
            if o_time > d_time:
                max_time_interval = d_time + (24-o_time)
            else:
                max_time_interval = d_time - o_time + (np.random.rand()-0.5)*np.random.rand()*0.8  #0.5 时间间隔容错   #卡达到时间 OD+Time   #一个hour抽一次
            if seg_period > max_time_interval:
                #print('Time period too long,pass.')
                continue
            #b.如果总耗时小于OD时间，则尝试插入
            for i in range(len(temp_detail_traj_list)):  #通过循环将轨迹写入总list中
                time = d_time -seg_period + temp_detail_traj_list[i][0] / 3600
                if time > 0:
                    detail_traj_list.append(
                        {'day':day,
                         'uid': uid, 'time': time,
                         'lat': temp_detail_traj_list[i][1],
                         'lon': temp_detail_traj_list[i][2]})
                else:
                    detail_traj_list.append(
                        {'day': day-1,
                         'uid': uid, 'time': 24 + time,
                         'lat': temp_detail_traj_list[i][1],
                         'lon': temp_detail_traj_list[i][2]})
    return detail_traj_list

def lstmGen(dataset, model, text, next_words=1000,temperature=0.2):
    model.eval()

    words = text.split(' ')
    state_h, state_c = model.init_state(len(words))

    for i in range(0, next_words):
        x = torch.tensor([[dataset.word_to_index[w] for w in words[i:]]])
        y_pred, (state_h, state_c) = model(x, (state_h, state_c))

        last_word_logits = y_pred[0][-1]
        p = torch.nn.functional.softmax(last_word_logits, dim=0).detach().numpy()
        #word_index = np.random.choice(len(last_word_logits), p=p)
        word_index = wordSample(p, temperature= temperature)
        words.append(dataset.index_to_word[word_index])

        if (dataset.index_to_word[word_index] == 'DAY_END'):
            return words



def main(k):
    #=============================================Start Generation===============================================#

    #Experiment Settings
    ORI_LP_FOLDER = '../data/LifePattern/2_great_tokyo_labeled_home_work_order/'
    #r'F:\life_pattern_detect\great_tokyo\2_great_tokyo_labeled_home_work_order'
    OD_LIB_PATH = '../data/2013/2013_segment_all_great_tokyo/'
    #'F:/TrajData/2013/2013_segment_all_great_tokyo/'
    MODEL_TYPE = 'FNN'
    DATA_STRUCT_MODE ='vector'   # vector/matrix/vector_lstm
    LOC_MODE = 'Pure Gaussian Noise'  #With Location Info/ Pure Gaussian Noise
    TRAIN_USER_NUM = 5000
    HOLIDAY_MODE = 0
    #MODEL_FOLDER_PATH = '../assets/' + MODEL_TYPE + '_' + LOC_MODE +'/weekdaymode_'+ \
    #            str(1-HOLIDAY_MODE) + '_user_' + str(TRAIN_USER_NUM) +'/' + DATA_STRUCT_MODE + '/'

    #1.先分weekdays和holidays生成lp_code，2.再把lp_code接起来生成traj
    temperature = 0.1

    WeekTrajGenerator = TrajGenerator(MODEL_TYPE,LOC_MODE,DATA_STRUCT_MODE,TRAIN_USER_NUM,HOLIDAY_MODE,ORI_LP_FOLDER,OD_LIB_PATH,temperature)
    HOLIDAY_MODE = 1
    HoliTrajGenerator = TrajGenerator(MODEL_TYPE,LOC_MODE,DATA_STRUCT_MODE,TRAIN_USER_NUM,HOLIDAY_MODE,ORI_LP_FOLDER,OD_LIB_PATH,temperature)

    path_ = '../results/gene_20211102_traj_MM_Extensive/'
    if not os.path.exists(path_):
        os.mkdir(path_)
    elif os.path.exists(path_):
        if os.path.exists(path_ + 'logs_{}.txt'.format(k)):
            return
    #total_momth_fake_traj_list = []
    #text = "process #{}".format(k)
    for i in tqdm(range(k)):  #如果 len(lp_file_list)>OB_LIB_USER_NUM,则为len(lp_file_list)

        #Gene 5 weekdays
        month_weekday_lp_code_list = []
        uid = i

        file = path_ + 'detail_fake_traj_' + str(uid) + '_t02_1028.csv'
        if os.path.exists(file):
            continue

        G = WeekTrajGenerator.G
        #temp_lp_df = pd.read_csv(WeekTrajGenerator.lp_file_list[i % WeekTrajGenerator.OB_LIB_USER_NUM])
        #temp_lp_location_dict = lpFile2LocDict(temp_lp_df) # significant places
        #od_lib_dict = MiniTools.loadPKL(WeekTrajGenerator.od_lib_dict_file_list[i % WeekTrajGenerator.OB_LIB_USER_NUM])
        #od_lib_dict = dict(ChainMap(*od_lib_dict))

        if MODEL_TYPE == 'LSTM':
            fake_lstm_lp_list = []
            while(len(fake_lstm_lp_list)<3):
                fake_lstm_lp_list = lstmGen(WeekTrajGenerator.lstm_dataset, G.cpu(), text='0.H_0',temperature = temperature)
        else:
            #z = createRandomNoise(LOC_MODE, loc_info_list[i] / 20000000)
            z = WeekTrajGenerator.createRandomNoise(LOC_MODE, None)
            fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
            fake_images = WeekTrajGenerator.vector2Matrix(fake_img.cpu().data)

        if DATA_STRUCT_MODE =='vector':
            if MODEL_TYPE == 'GCN':
                temp_lp = DataFrame(fake_images.numpy()).T
            elif MODEL_TYPE == 'FNN':
                temp_lp = DataFrame(fake_images.numpy())
            temp_lp.columns = WeekTrajGenerator.vector_form_df.columns
        elif DATA_STRUCT_MODE =='matrix':
            temp_lp = DataFrame(minitools.normSumAxis1(fake_images.numpy()[0][0]))
            temp_lp.columns = WeekTrajGenerator.matrix_form_df.columns
        elif DATA_STRUCT_MODE =='vector_lstm':
            temp_lp = fake_lstm_lp_list


        for m in range(4):
            week_weekday_lp_code_list = []
            for n in range(5):
                day_lp_code_list = lpPob2LpCode(temp_lp, DATA_STRUCT_MODE, WeekTrajGenerator.vector_form_df,WeekTrajGenerator.temperature)
                week_weekday_lp_code_list = week_weekday_lp_code_list + day_lp_code_list
            # DataFrame(week_weekday_lp_code_list).to_csv(
            #     'F:/TrajData/2013/gene_20211023/' + 'weekday_lp_code_list_week' +str(m+1) + '_'  +str(uid) + '_t02_.csv')
            month_weekday_lp_code_list.append(week_weekday_lp_code_list)

        # Gene 2 holidays
        month_holiday_lp_code_list = []
        uid = i

        G = HoliTrajGenerator.G
        temp_lp_df = pd.read_csv(HoliTrajGenerator.lp_file_list[i % WeekTrajGenerator.OB_LIB_USER_NUM])
        temp_lp_location_dict = lpFile2LocDict(temp_lp_df)  # significant places
        od_lib_dict = minitools.loadPKL(HoliTrajGenerator.od_lib_dict_file_list[i % HoliTrajGenerator.OB_LIB_USER_NUM])
        od_lib_dict = dict(ChainMap(*od_lib_dict))

        if MODEL_TYPE == 'LSTM':
            fake_lstm_lp_list = []
            while (len(fake_lstm_lp_list) < 3):
                fake_lstm_lp_list = lstmGen(HoliTrajGenerator.lstm_dataset, G.cpu(), text='0.H_0',temperature = temperature)
        else:
            z = HoliTrajGenerator.createRandomNoise(LOC_MODE, None)
            fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
            fake_images = HoliTrajGenerator.vector2Matrix(fake_img.cpu().data)

        if DATA_STRUCT_MODE == 'vector':
            if MODEL_TYPE == 'GCN':
                temp_lp = DataFrame(fake_images.numpy()).T
            elif MODEL_TYPE == 'FNN':
                temp_lp = DataFrame(fake_images.numpy())
            temp_lp.columns = HoliTrajGenerator.vector_form_df.columns
        elif DATA_STRUCT_MODE == 'matrix':
            temp_lp = DataFrame(minitools.normSumAxis1(fake_images.numpy()[0][0]))
            temp_lp.columns = HoliTrajGenerator.matrix_form_df.columns
        elif DATA_STRUCT_MODE == 'vector_lstm':
            temp_lp = fake_lstm_lp_list

        for m in range(4):
            week_holiday_lp_code_list = []
            for n in range(2):
                day_lp_code_list = lpPob2LpCode(temp_lp, DATA_STRUCT_MODE, HoliTrajGenerator.vector_form_df,HoliTrajGenerator.temperature)
                week_holiday_lp_code_list = week_holiday_lp_code_list + day_lp_code_list
            # DataFrame(week_holiday_lp_code_list).to_csv(
            #     'F:/TrajData/2013/gene_20211023/' + 'holiday_lp_code_list_week' +str(m+1) + '_' + str(uid) + '_t02_.csv')
            month_holiday_lp_code_list.append(week_holiday_lp_code_list)

        momth_lp_code_list = []
        for m in range(4):
            week_lp_code_list = month_weekday_lp_code_list[m] + month_holiday_lp_code_list[m]
            momth_lp_code_list = momth_lp_code_list + week_lp_code_list

        detail_gene_traj = lpCode2Traj(uid, momth_lp_code_list, temp_lp_location_dict, od_lib_dict)

        detail_gene_traj_df = DataFrame(detail_gene_traj)
        #2013/7/1是星期一
        try:
            detail_gene_traj_df['str_time'] = ["2013-7-%d %02d:%02d:00"%(day+1,int(time),int((time-int(time))*60)) for day,time in detail_gene_traj_df[['day','time']].values]

            detail_gene_traj_df.to_csv(path_ + 'detail_fake_traj_'+str(uid) +  '_t02_1028.csv')

            #MiniTools.savePKL(momth_lp_code_list,'F:/TrajData/2013/gene_20211023 - traj/' + 'fake_lp_code_'+str(uid) +  '_t02_.pkl')

            #total_momth_fake_traj_list.append(detail_gene_traj_df)
            #if i%500 == 0:
            #    total_momth_fake_traj_df = pd.concat(total_momth_fake_traj_list)
            #    total_momth_fake_traj_df.to_csv('../..//2013/' + 'total_momth_fake_traj_0r_MM.csv')
        except:
            #print('None has been generated.')
            continue

    f = open(path_ + 'logs_{}.txt'.format(k), mode='w')
    f.write('finished...')
    f.close()

    return

if __name__ == '__main__':

    start_time = time.time()

    #p = Pool(processes=1)
    #sample_size = 1
    #for _ in tqdm(p.imap_unordered(main, range(int(sample_size))), total=int(sample_size)):
    #    pass

    #p.close()
    #p.join()
    main(100)
    end_time = time.time()
    print(end_time - start_time)
