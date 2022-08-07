# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for trajectory pre-processing'
__author__ = 'Li Peiran'

# 3.Import the modules.
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
#from tqdm.notebook import tqdm
from skmob.measures.individual import home_location
import multiprocessing as mp
import threading as td
import time
import pickle
import jismesh.utils as ju
import pandas as pd
from pandas import DataFrame
from scipy import sparse
# For NN
import torch
import torch.nn.functional as F  # 激励函数都在这
from torch.autograd import Variable
import torch.nn as nn
import scipy.stats
from pyswarm import pso
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
from utils import MiniTools
import datetime


# 4.Define the global variables. (if exists)
SAVE_PATH = '../assets/LSTM_None/'
# 5.Define the class (if exsists)

# 6.Define the function (if exsists)


#=================================Load the total cases to form the matrix shape==========================================#
LP_CODE_FILE = r'F:\life_pattern_detect\great_tokyo\4_tree_index\total_tree_index.csv'
lp_code_df = pd.read_csv(LP_CODE_FILE)
lp_code_dict = {}
for time,temp_df in lp_code_df.groupby('time'):
    current_lp_code_dict = {}
    # print(len(temp_df))
    for i in temp_df.index.values:
        #temp_df = temp_df.reset_index()
        hour = temp_df.loc[i,'time']
        place = temp_df.loc[i,'places']
        next_place = temp_df.loc[i,'next_places']
        temp_dict = {( place + '.' + next_place): temp_df.loc[i,'tree_index']}
        current_lp_code_dict.update(temp_dict)
    lp_code_dict.update({time:current_lp_code_dict})

total_lp_code_df = DataFrame(lp_code_dict).T
total_lp_code_df.loc[:,:]=0

#Save all the life-pattern code and time pandas (format base)
#total_lp_code_df.to_csv('LP_format.csv')

#=================================Calculate Life Pattern from Original HWO files==========================================#
GENE_MODE = 'Re_Calculate'

ORI_LP_FOLDER = r'F:\life_pattern_detect\great_tokyo\2_great_tokyo_labeled_home_work_order'
lp_file_list = []
MiniTools.getFilePath(ORI_LP_FOLDER, lp_file_list, dir_list=[], target_ext='.csv')

if GENE_MODE == 'Re_Calculate':
    total_lp_code_list = []
    for i in tqdm(range(40)):
        temp_total_lp_code_df = total_lp_code_df.copy()
        # 1.Get the hour state
        current_lp_df = pd.read_csv(lp_file_list[i])
        current_lp_df = current_lp_df[current_lp_df['holiday']==0].reset_index() #filter holidays
        hour_state_list = []
        start_flag = False
        for j in range(1,len(current_lp_df)):
            hour = current_lp_df.loc[j, 'hour']
            endhour = current_lp_df.loc[j, 'endhour']
            all_detect_label = current_lp_df.loc[j, 'all_detect_label']
            home_label = current_lp_df.loc[j, 'home_label_order']
            work_label = current_lp_df.loc[j, 'work_label_order']
            other_label = current_lp_df.loc[j, 'other_label_order']
            if all_detect_label > -1:
                start_flag = True
            if start_flag == False:
                continue
            if home_label > -1:
                for j in range(hour, endhour):
                    # if len(hour_state_list)>0:
                    #     if ('H_' + str(home_label)) != hour_state_list[-1].split('.')[1]:
                    #         hour_state_list.append((str(j) + '.H_' + str(home_label)))
                    # else:
                    hour_state_list.append((str(j) + '.H_' + str(home_label)))
            elif work_label > -1:
                for j in range(hour, endhour):
                    # if len(hour_state_list) > 0:
                    #     if ('W_' + str(work_label)) != hour_state_list[-1].split('.')[1]:
                    #         hour_state_list.append((str(j) + '.W_' + str(work_label)))
                    # else:
                    hour_state_list.append((str(j) + '.W_' + str(work_label)))
            elif other_label > -1:
                for j in range(hour, endhour):
                    # if len(hour_state_list) > 0:
                    #     if ('O_' + str(other_label)) != hour_state_list[-1].split('.')[1]:
                    #         hour_state_list.append((str(j) + '.O_' + str(other_label)))
                    # else:
                    hour_state_list.append((str(j) + '.O_' + str(other_label)))

        #2. From hour state to lp code
        for k in range(len(hour_state_list) - 1):
            if (hour_state_list[k]!='DAY_END'):
                hour = int(hour_state_list[k].split('.')[0])
                next_hour = int(hour_state_list[k+1].split('.')[0])
                if (hour>next_hour):
                    hour_state_list.insert(k+1, 'DAY_END')
                    if hour_state_list[k + 2]!='0.H_0':
                        hour_state_list.insert(k + 2, '0.H_0')

        # for k in range(len(hour_state_list) - 1):
        #     hour = int(hour_state_list[k].split('.')[0])


        total_lp_code_list = total_lp_code_list + hour_state_list
        #DataFrame(hour_state_list).to_csv(SAVE_PATH + 'temp_one_user.csv')

    DataFrame(total_lp_code_list).to_csv(SAVE_PATH + 'user_40.csv')

# elif GENE_MODE == 'LOAD_EXISTING':
#     total_lp_code_list = MiniTools.loadPKL('2000_input_lp_list.pkl')
