# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for dataloader'
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
import datetime
from utils import MiniTools
from models import FNNGAN
from models import DCGAN
from models import GCN
import torch.autograd
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
#from models.GCN import GCN

# 4.Define the global variables. (if exists)

def vector2Matrix(x,DATA_STRUCT_MODE):
    if DATA_STRUCT_MODE =='matrix':
        out = x.view(-1, 1, image_size[0], image_size[1])
        return out
    elif DATA_STRUCT_MODE =='vector':
        out = x
        return out
    return 0

def createRandomNoise(mode,add_info):
    z = 0
    if mode=='With Location Info':
        z_loc = Variable(torch.tensor(add_info, dtype=torch.float32)).to(device).reshape(1, -1)
        z_random = Variable(torch.randn(batch_size, len(z_loc[0]))).to(device)  # 得到随机噪声
        z = z_random + z_loc
    elif mode=='Pure Gaussian Noise':
        z = Variable(torch.randn(batch_size, z_dimension)).to(device)  # for FNN
    else:
        print('Please Give Correct Mode Name.')
    return z

def createRandomNoise_1():
    z = Variable(torch.randn(2, z_dimension)).to(device)
    return z

def renewGraph(graph,node_embed,temp_edges):
    #1. renew node
    graph.x = torch.LongTensor(node_embed).to(device)
    #2. renew edge
    edges = []
    for i in range(len(temp_edges)):
        o_index = temp_edges[i][0][0]
        d_index = temp_edges[i][1][0]
        edges.append([o_index,d_index])
    graph.edge_index = torch.tensor(edges).T.long().to(device)

    return graph

def renewLoc_info(input_kp_list):
    loc_info = pd.Series(data=np.zeros(len(key_point_pool)),index=key_point_pool)
    for key in input_kp_list.keys():
        loc_info[key] = input_kp_list[key]
    loc_info = Variable(torch.tensor(loc_info.values).reshape(-1)).to(device) #, dtype=torch.float32
    return  loc_info

def hourNorm(kp_format_df):
    kp_format_series = pd.Series(data=range(len(kp_format_df)), index=kp_format_df.index)
    kp_format_matrix_df = kp_format_series.unstack().T
    hour_norm_group_list = []
    for i in range(len(kp_format_matrix_df.values[:, 0])):
        x = kp_format_matrix_df.values[i]
        x = x[~np.isnan(x)]
        hour_norm_group_list.append(x)
    return hour_norm_group_list
