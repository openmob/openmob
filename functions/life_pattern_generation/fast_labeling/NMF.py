import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime
import random
import natsort
from sklearn.decomposition import NMF
import sklearn.cluster
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq
from scipy.optimize import least_squares
from tqdm import tqdm


def Fun(p, meta):  # 定义拟合函数形式
    x, y, z = p
    meta1,meta2,meta3 = meta
    LP_simulate = x * meta1  + y * meta2 + z * meta3

    return LP_simulate


def error(p, meta, LP_actual):  # 拟合残差;这里之所以没有平方是应为在拟合函数的内部进行，这里不显式的表示。
    return Fun(p, meta) -  LP_actual
#
#
# def main():
#

def lpVector2treeIndex(total_lp_wgangp_list,lp_format):

    # 1. lpVector2treeIndex
    module_path = os.path.dirname(__file__)
    TREE_INDEX_PATH =  module_path + '/the_great_tokyo_tree_index_multiple_HW_single_O.csv'
    tree_index_df = pd.read_csv(TREE_INDEX_PATH)

    new_cols = []
    for x in lp_format:
        o_place = x[0].split('.')[0]
        d_place = x[0].split('.')[1]
        if 'O' in o_place:
            o_place = 'O'
        if 'O' in d_place:
            d_place = 'O'
        new_cols.append(o_place + '.' + d_place + str(x[1]))
    new_cols = np.array(new_cols)

    trans_indexs = []
    for i in tree_index_df.index:
        hour = tree_index_df.loc[i, 'time']
        place = tree_index_df.loc[i, 'places']
        next_place = tree_index_df.loc[i, 'next_places']
        col_name = place + '.' + next_place

        cr_index = np.where(new_cols == col_name + str(hour))
        #     print(new_cols)
        #     print(col_name + str(hour))
        #     print(cr_index)
        # print(test_wgan_lp['od_hour'].values[cr_index].sum())
        trans_indexs.append(cr_index)

    total_tree_index_list = []
    print('From Life Pattern Matrix/Vector to Life Pattern Tree Index Value')
    for user in tqdm(range(0, len(total_lp_wgangp_list))):
        # 1.
        pob_list = total_lp_wgangp_list[user]
        tree_index_list = np.zeros(len(tree_index_df))
        for i in range(len(tree_index_df)):
            tree_index_list[i] = pob_list[trans_indexs[i]].sum()
            # print(tree_index_list[i])

        total_tree_index_list.append(tree_index_list)
        # tree_index_df['0'] = tree_index_list
        # tree_index_df['0'].to_csv('./temp_tree_index_files/lp_tree_index_%d_test.csv' % user)
    return total_tree_index_list

def nfm(total_tree_index_list):
    module_path = os.path.dirname(__file__)
    df_meta = pd.read_csv(module_path + '/2013_W_multiple_HW_single_O_total_day.csv')
    df_meta.columns = ['meta1', 'meta2', 'meta3']
    meta1 = np.array(df_meta['meta1'])
    meta2 = np.array(df_meta['meta2'])
    meta3 = np.array(df_meta['meta3'])

    x = [meta1, meta2, meta3]
    p0 = [0.01, 0.01, 0.01]  ##


    #runlist = []

    # for root, dirs, files in os.walk(individual_life_pattern_dir):
    #     for filename in files:
    #         if filename.endswith('.csv'):
    #             filepath = os.path.join(root, filename)
    #             name = filename.split('_')
    #             runlist.append((filepath,name[0]))
    #             print('filename:', name[0])


###########################################################
    result_record_list = []
    print('From Life Pattern Tree Index Value to XYZ list (by NFM)')
    for i in tqdm(range(len(total_tree_index_list))):
        # df_individual = pd.read_csv(runlist[i][0])
        # user_ID = runlist[i][1]
        y = total_tree_index_list[i]
        para = least_squares(error, p0, args=(x, y),bounds = (0,np.inf))  # 进行拟合;args() 中是除了初始值之外error() 中的所有参数的集合输入。
        xyz = para.x
        result_record_list.append([i, xyz[0], xyz[1], xyz[2]])

    result = pd.DataFrame(result_record_list, columns=['gene_user_ID', 'X', 'Y', 'Z'])

    return result[['X', 'Y', 'Z']].values




