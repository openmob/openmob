# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for some utilization tools'
__author__ = 'Li Peiran'

# 3.Import the modules.
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
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


# 4.Define the global variables. (if exists)

# 5.Define the class (if exsists)

# 6.Define the function (if exsists)


def getFilePath(root_path, file_list, dir_list=[], target_ext=[]):
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            getFilePath(dir_file_path, file_list, dir_list, target_ext)
        else:
            ext = os.path.splitext(dir_file_path)[1]  # 获取后缀名
            if ext == target_ext:
                file_list.append(dir_file_path)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def normMinMaxAxis1(data):
    _range = np.max(data, axis=1) - np.min(data, axis=1)
    return (data - np.min(data, axis=1)[:, None]) / _range[:, None]

def normSum(data):
    _sum = np.sum(data)
    return np.nan_to_num(data / _sum) #trans the nan to zero


def normSumAxis1(data):
    _sum = np.sum(data, axis=1)
    return np.nan_to_num(data / _sum[:, None]) #trans the nan to zero

def normSum_Tensor(data):
    _sum = data.sum()
    return data / _sum #trans the nan to zero

def normSumAxis1_Tensor(data):
    _sum = data.sum(axis=1)
    return data / _sum[:, None] #trans the nan to zero



def numpyMSE(arr1, arr2):
    return np.square(np.subtract(arr1, arr2)).mean()


def numpyMAE(arr1, arr2):
    return np.abs(np.subtract(arr1, arr2)).mean()


def ifFolderExistThenCreate(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print('Create Folder: %s' % dir)
    return 1


def getGaussianPob(mu, std, value):
    return scipy.stats.norm(mu, std).pdf(value)


def getGaussianPobTensor(mean, std, value):
    return 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-(torch.pow((value - mean), 2) / 2 / std / std))


def savePKL(obj, name):
    """
    Save data as pickle file.
    """
    if name[-4:] == '.pkl':
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def loadPKL(name):
    """
    Load data from pickle file.
    """
    with open(name, 'rb') as f:
        return pickle.load(f)

import pandas as pd
import os
import chardet
def get_encoding(filename):
    """
    返回文件编码格式
    """
    with open(filename,'rb') as f:
        return chardet.detect(f.read())['encoding']

def normRange(data,range=(-1,1)):
    from sklearn.preprocessing import MinMaxScaler
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=range)
        scaler.fit(data) # scaler.data_max_
        data = scaler.transform(data)
        return data[:, 0]
    else:
        scaler = MinMaxScaler(feature_range=range)
        scaler.fit(data) # scaler.data_max_
        data = scaler.transform(data)
        return data
