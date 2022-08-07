# !/usr/bin/env python
# -*- coding: utf-8 -*-


"""This file is for trajectory pre-processing"""
__author__ = 'Li Peiran'

import chardet
import os
import pickle
import torch
import scipy
import numpy as np


def get_file_path(root_path, file_list, dir_list=None, target_ext=None):
    # 获取该目录下所有的文件名称和目录名称
    if target_ext is None:
        target_ext = []
    if dir_list is None:
        dir_list = []
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            dir_list.append(dir_file_path)
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path, file_list, dir_list, target_ext)
        else:
            ext = os.path.splitext(dir_file_path)[1]  # 获取后缀名
            if ext == target_ext:
                file_list.append(dir_file_path)


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def norm_min_max_axis1(data):
    _range = np.max(data, axis=1) - np.min(data, axis=1)
    return (data - np.min(data, axis=1)[:, None]) / _range[:, None]


def norm_sum_axis1(data):
    _sum = np.sum(data, axis=1)
    return np.nan_to_num(data / _sum[:, None])  # trans the nan to zero


def norm_sum_axis1_tensor(data):
    _sum = data.sum(axis=1)
    return data / _sum[:, None]  # trans the nan to zero


def numpy_mse(arr1, arr2):
    return np.square(np.subtract(arr1, arr2)).mean()


def numpy_mae(arr1, arr2):
    return np.abs(np.subtract(arr1, arr2)).mean()


def if_folder_exist_then_create(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print('Create Folder: %s' % directory)
    return 1


def get_gaussian_pob(mu, std, value):
    return scipy.stats.norm(mu, std).pdf(value)


def get_gaussian_pob_tensor(mean, std, value):
    return 1 / (std * np.sqrt(2 * np.pi)) * torch.exp(-(torch.pow((value - mean), 2) / 2 / std / std))


def save_pkl(obj, name):
    """
    Save data as pickle file.
    """
    if name[-4:] == '.pkl':
        with open(name, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_pkl(name):
    """
    Load data from pickle file.
    """
    with open(name, 'rb') as f:
        return pickle.load(f)


def norm_sum(data):
    _sum = np.sum(data)
    return np.nan_to_num(data / _sum)  # trans the nan to zero


def get_encoding(filename):
    """
    返回文件编码格式
    """
    with open(filename, 'rb') as f:
        return chardet.detect(f.read())['encoding']
