import numpy as np
import jismesh.utils as ju
import torch
import torch_geometric.datasets as datasets
import torch_geometric.data as data
import torch_geometric.transforms as transforms
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt
from utils import MiniTools
import pandas as pd
from pandas import DataFrame
from pyvis.network import Network
from tqdm.notebook import tqdm
from models import GS2P
from utils import LPUtils
from torch import nn, optim
from torch.utils.data import DataLoader
import seaborn as sns
from collections import ChainMap
import geopandas as gpd
import transbigdata as tbd

MESH_DEGREE = 3
demo_group = ['0-15*M','0-15*F','15-20*M','15-20*F', '20-30*M','20-30*F','30-40*M','30-40*F','40-50*M','40-50*F',
              '50-60*M','50-60*F','60-70*M','60-70*F','70-80*M','70-80*F']

# 计算全部H_0-mesh对应的demo
# DEMO_INFO_PATH = r'C:\Users\lipeiran\OneDrive - The University of Tokyo\Income Labelling\ForYqSystem\Filter30_16groups_MAE_4week_VI_choose.csv'
# demo_df = pd.read_csv(DEMO_INFO_PATH,index_col=0)
# demo_df['uid'] = demo_df.index

#Get the user order list
USER_NO_PATH = r'F:\TrajData\Traj_Generation_Integration\Scaling_Factor\ScalingFactor.xlsx'
user_no = pd.read_excel(USER_NO_PATH)['user'].values
#Construt a key point dict
KEY_POINT_PATH = r'F:\TrajData\Traj_Generation_Integration\True_Key_Point\74693_key_point_lonlat.pkl'
kps = MiniTools.loadPKL(KEY_POINT_PATH)

# 获得全部可能的loc
all_locs = []
for user in kps.keys():
    cr_kp = kps[user]
    key = value = 0
    for loc in cr_kp.keys():
        all_locs.append(loc)
all_locs = np.unique(np.array(all_locs))

random_kp_pob_dict = {}
uids = ['%08d' %uid for uid in user_no]
for CR_LOC in all_locs:
    CR_LOC_mesh_list = []
    for uid in uids:
        try:
            mesh = ju.to_meshcode(kps[uid][CR_LOC][0],kps[uid][CR_LOC][1],3)
            CR_LOC_mesh_list.append(mesh)
        except:
            CR_LOC_mesh_list.append(0)
    unique, counts = np.unique(CR_LOC_mesh_list, return_counts=True)
    random_kp_pob_dict.update({CR_LOC: (unique, counts / np.sum(counts))})

MiniTools.savePKL(all_locs,'../KeyPoint_PobTable/Home_Mesh_Demo_Table/all_locs.pkl')
MiniTools.savePKL(random_kp_pob_dict,'../KeyPoint_PobTable/Home_Mesh_Demo_Table/kp_random_pob_dict.pkl')


# #Construt a key point dict
# KEY_POINT_PATH = r'F:\TrajData\Traj_Generation_Integration\True_Key_Point\74693_key_point_lonlat.pkl'
# kps = MiniTools.loadPKL(KEY_POINT_PATH)
#
# # 获得全部可能的loc
# all_locs = []
# for user in kps.keys():
#     cr_kp = kps[user]
#     key = value = 0
#     for loc in cr_kp.keys():
#         if loc == 'H_0' or loc == 'W_0':
#             pass
#         else:
#             all_locs.append(loc)
# all_locs = np.unique(np.array(all_locs))
#
# h_0_list = []
# w_0_list = []
# hw_list = []
# all_others_list = {}
# for i in range(len(all_locs)):
#     all_others_list.update({all_locs[i]:[]})
# # H-table H->W Table
# kp_stat_list = []
# kps_list = list(kps.values())
# for i in range(len(kps_list)):
#     cr_kps = kps_list[i]
#     mesh_degree = MESH_DEGREE
#     keys = list(cr_kps.keys())
#     H_0_mesh = ju.to_meshcode(cr_kps['H_0'][0], cr_kps['H_0'][1], mesh_degree)
#     h_0_list.append(H_0_mesh)
#     if 'W_0' in keys:
#         W_0_mesh = ju.to_meshcode(cr_kps['W_0'][0], cr_kps['W_0'][1], mesh_degree)
#     else:
#         W_0_mesh = 0
#     w_0_list.append(W_0_mesh)
#     HW_mesh = str(H_0_mesh) + '_' + str(W_0_mesh)
#     hw_list.append(HW_mesh)
#     for loc_type in all_locs:
#         if loc_type in keys:
#             cr_loc_mesh = ju.to_meshcode(cr_kps[loc_type][0], cr_kps[loc_type][1], mesh_degree)
#         else:
#             cr_loc_mesh = 0
#         all_others_list[loc_type].append(cr_loc_mesh)
# # H-> W
# h2w_df = DataFrame(index = h_0_list,data = w_0_list,columns=['w_0_mesh'])
# h2w_pob_dict = {}
# for key, value in h2w_df.groupby(h2w_df.index)['w_0_mesh']:
#     unique, counts = np.unique(value, return_counts=True)
#     h2w_pob_dict.update({key:(unique,counts/np.sum(counts))})
# MiniTools.savePKL(h2w_pob_dict,'../KeyPoint_PobTable/h2w/'+ 'h2w_pob_dict.pkl')
# # HW -> Others
# for CR_LOC in all_locs:
#     cr_o_list = all_others_list[CR_LOC]
#     hw2o_df = DataFrame(index = hw_list,data = cr_o_list,columns=[CR_LOC])
#     hw2o_pob_dict = {}
#     for key, value in hw2o_df.groupby(hw2o_df.index)[CR_LOC]:
#         unique, counts = np.unique(value, return_counts=True)
#         hw2o_pob_dict.update({key:(unique,counts/np.sum(counts))})
#     MiniTools.savePKL(hw2o_pob_dict,'../KeyPoint_PobTable/Others/'+ 'hw_'+ CR_LOC+'_pob_dict.pkl')
#
#
#
# # kp_stat_list.append({'user':i,'key_type':key,'loc_mesh':loc_mesh})
# # kp_stat_df = DataFrame(kp_stat_list)
# # # h.w 有的占比，算出来h是1，说明每个人都有h_0
# # p_has_h0 = len(kp_stat_df[kp_stat_df['key_type']=='H_0'])/len(list(kps.values()))
# # p_has_w0 = len(kp_stat_df[kp_stat_df['key_type']=='W_0'])/len(list(kps.values()))*p_has_h0
# # #统计H_0->W_0概率统计表
# # hw_stat_list = []
# # for index,x in kp_stat_df.groupby('user'):
# #     #print(index)
# #     h_mesh = x[x['key_type']=='H_0']['loc_mesh'].values[0]
# #     if len(x[x['key_type']=='W_0']) > 0:
# #         w_mesh = x[x['key_type']=='W_0']['loc_mesh'].values[0]
# #     else:
# #         w_mesh = 0
# #     hw_stat_list.append({'h_mesh':h_mesh,'w_mesh':w_mesh})
# # hw_stat_df = DataFrame(hw_stat_list)
# # #统计H_0情况下，W_0的分布
# # unique = np.unique(hw_stat_df['h_mesh'].values)
# # h2w_pob_dict = {}
# # hw_combinations = []
# # for h0 in unique:
# #     w0_list = hw_stat_df[hw_stat_df['h_mesh'] == h0]['w_mesh'].values
# #     unique, counts = np.unique(w0_list, return_counts=True)
# #     h2w_pob_dict.update({h0:(unique,counts/np.sum(counts))})
# #     for w0 in w0_list:
# #         hw_combinations.append(str(h0) + '_' + str(w0))
# # MiniTools.savePKL(h2w_pob_dict,'h2w_pob_dict.pkl')
# #
# # # 获得全部可能的loc
# # all_locs = []
# # for user in kps.keys():
# #     cr_kp = kps[user]
# #     key = value = 0
# #     for loc in cr_kp.keys():
# #         if loc == 'H_0' or loc == 'W_0':
# #             pass
# #         else:
# #             all_locs.append(loc)
# # all_locs = np.unique(np.array(all_locs))
# #
# # #统计H_0->其他LOC概率统计表
# # for CR_LOC in all_locs:
# #     hw2o_stat_list = []
# #     for index,x in kp_stat_df.groupby('user'):
# #         #print(index)
# #         h_mesh = x[x['key_type']=='H_0']['loc_mesh'].values[0]
# #         if len(x[x['key_type']=='W_0']) > 0:
# #             w_mesh = x[x['key_type']=='W_0']['loc_mesh'].values[0]
# #         else:
# #             w_mesh = 0
# #         hw_comb = str(h_mesh) + '_' + str(w_mesh)
# #         if len(x[x['key_type']==CR_LOC]) > 0:
# #             o_mesh = x[x['key_type']==CR_LOC]['loc_mesh'].values[0]
# #         else:
# #             o_mesh = 0
# #
# #         hw2o_stat_list.append({'hw_comb':hw_comb,CR_LOC+'_mesh':o_mesh})
# #     hw2o_stat_df = DataFrame(hw2o_stat_list)
# #     #统计HW情况下，all O的分布
# #     unique = np.unique(hw2o_stat_df['hw_comb'].values)
# #     hw2o_pob_dict = {}
# #     hw_combinations = []
# #     for hw in unique:
# #         o_list = hw2o_stat_df[hw2o_stat_df['hw_comb'] == hw][CR_LOC+'_mesh'].values
# #         unique, counts = np.unique(o_list, return_counts=True)
# #         hw2o_pob_dict.update({hw:(unique,counts/np.sum(counts))})
# #     #hw2o_pob_df = DataFrame(hw2o_pob_list)
# #     MiniTools.savePKL(hw2o_pob_dict,'hw_'+ CR_LOC+'_pob_dict.pkl')