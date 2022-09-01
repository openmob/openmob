# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
"""This file is for trajectory pre-processing as the life-pattern model input"""
__author__ = 'Li Peiran'

# 3.Import the modules.
import jismesh.utils as ju
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
import minitools
import numpy as np

class InputGenerationFromLabelTraj:
    def __init__(self, lp_code_file, lp_format_output_file, gene_mode, week_holiday_mode, ori_lp_folder):
        """
        lp_code_file = r'F:\life_pattern_detect\great_tokyo\4_tree_index\total_tree_index.csv'
        lp_format_output_file = 'LP_format.csv'
        gene_mode = Re_Calculate / LOAD_EXISTING
        ori_lp_folder = r'F:\life_pattern_detect\great_tokyo\2_great_tokyo_labeled_home_work_order'

        """
        self.lp_code_file = lp_code_file
        self.lp_format_output_file = lp_format_output_file
        self.gene_mode = gene_mode
        self.week_holiday_mode = week_holiday_mode
        self.ori_lp_folder = ori_lp_folder

    def lp_format(self):  # file path need  further modification
        """
        Load  the total cases to form the matrix shape
        :return: None
        """

        lp_code_df = pd.read_csv(self.lp_code_file)
        lp_code_dict = {}
        for time, temp_df in lp_code_df.groupby('time'):
            current_lp_code_dict = {}
            # print(len(temp_df))
            for i in temp_df.index.values:
                # temp_df = temp_df.reset_index()
                hour = temp_df.loc[i, 'time']
                place = temp_df.loc[i, 'places']
                next_place = temp_df.loc[i, 'next_places']
                temp_dict = {(place + '.' + next_place): temp_df.loc[i, 'tree_index']}
                current_lp_code_dict.update(temp_dict)
            lp_code_dict.update({time: current_lp_code_dict})

        total_lp_code_df = DataFrame(lp_code_dict).T
        total_lp_code_df.loc[:, :] = 0

        # Save all the life-pattern code and time pandas (format base)
        total_lp_code_df.to_csv(self.lp_format_output_file)
        self.total_lp_code_df = total_lp_code_df

    def lp_calculation(self):
        """
        Calculate life pattern from original HWO files
        """

        OUT_PUT_FOLDER = '../assets/Input_LP/' + str(1 - self.week_holiday_mode) + '/'  # 1->weekday  2->weekend
        OUT_PUT_FOLDER_INDIVIDUAL = 'F:/TrajData/Traj_Generation_Integration/True_Life_Pattern/'
        minitools.if_folder_exist_then_create(OUT_PUT_FOLDER)

        lp_file_list = []
        minitools.get_file_path(self.ori_lp_folder, lp_file_list, dir_list=[], target_ext='.csv')
        user_num = len(lp_file_list)

        if self.gene_mode == 'Re_Calculate':
            total_lp_code_list = []
            for i in tqdm(range(1)):
                temp_total_lp_code_df = self.total_lp_code_df.copy()
                # 1.Get the hour state
                current_lp_df = pd.read_csv(lp_file_list[i])
                current_lp_df = current_lp_df[
                    current_lp_df['holiday'] == self.week_holiday_mode].reset_index()  # filter holidays
                hour_state_list = []
                start_flag = False
                for j in range(len(current_lp_df)):
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
                            hour_state_list.append({j: 'H_' + str(home_label)})
                    elif work_label > -1:
                        for j in range(hour, endhour):
                            hour_state_list.append({j: 'W_' + str(work_label)})
                    elif other_label > -1:
                        for j in range(hour, endhour):
                            hour_state_list.append({j: 'O_' + str(other_label)})
                # 2. From hour state to lp code
                for k in range(len(hour_state_list) - 1):
                    (hour, state), = hour_state_list[k].items()
                    (next_hour, next_state), = hour_state_list[k + 1].items()
                    try:
                        temp_total_lp_code_df.loc[next_hour, state + '.' + next_state] = temp_total_lp_code_df.loc[
                                                                                             next_hour, state + '.' + next_state] + 1
                    except:
                        print('Life-pattern code not find in the total list! (skip)')
                temp_total_lp_code_df.to_csv(
                    OUT_PUT_FOLDER_INDIVIDUAL + 'true_life_pattern_' + lp_file_list[i].split('\\')[-1].split('_')[
                        0] + '.csv')
                total_lp_code_list.append(temp_total_lp_code_df.values)

            minitools.save_pkl(total_lp_code_list, OUT_PUT_FOLDER + str(user_num) + '_input_lp_list.pkl')
            return

        elif self.gene_mode == 'LOAD_EXISTING':
            total_lp_code_list = minitools.load_pkl(OUT_PUT_FOLDER + str(user_num) + '_input_lp_list.pkl')
            return total_lp_code_list

    def down_sampling(total_lp_code_list=None):

        # =================================Down Sampling to form Input==========================================#

        # Down-sampling
        down_total_lp_code_list = total_lp_code_list.copy()
        lp_score = []
        for i in tqdm(range(len(down_total_lp_code_list))):
            temp_down_sampling_lp_df = down_total_lp_code_list[i]
            down_rate = np.random.rand()
            # 随机置为0
            have_value_index = np.where(temp_down_sampling_lp_df > 0)
            delete_index = np.random.rand(len(have_value_index[0]))
            delete_index = (delete_index < down_rate).astype(int)
            # 赋值回去
            temp_down_sampling_lp_df[have_value_index] = temp_down_sampling_lp_df[have_value_index] * delete_index
            down_total_lp_code_list[i] = temp_down_sampling_lp_df
            # 保存分数
            lp_score.append(down_rate)

        minitools.save_pkl(down_total_lp_code_list, '2000_input_down_lp_list.pkl')
        np.save('2000_input_down_lp_score.npy', lp_score)
        return

    # ===============================Record the Location ===========================================#
    def lp_file2_loc_dict_meshcode(current_lp_df):
        home_place_num = len(current_lp_df['home_label_order'].unique()) - 1
        work_place_num = len(current_lp_df['work_label_order'].unique()) - 1
        other_place_num = len(current_lp_df['other_label_order'].unique()) - 1
        home_places = []
        work_places = []
        other_places = []
        for k in range(home_place_num):
            temp = current_lp_df[current_lp_df['home_label_order'] == k][['home_lat', 'home_lon']].iloc[0].values
            temp_mesh_code = ju.to_meshcode(temp[0], temp[1], 5)
            home_places.append(temp_mesh_code)
        for k in range(work_place_num):
            temp = current_lp_df[current_lp_df['work_label_order'] == k][['work_lat', 'work_lon']].iloc[0].values
            temp_mesh_code = ju.to_meshcode(temp[0], temp[1], 5)
            work_places.append(temp_mesh_code)
        for k in range(other_place_num):
            temp = current_lp_df[current_lp_df['other_label_order'] == k][['other_lat', 'other_lon']].iloc[0].values
            temp_mesh_code = ju.to_meshcode(temp[0], temp[1], 5)
            other_places.append(temp_mesh_code)
        # 2.建立统一结构的lp_code-location字典
        home_location_dict = {'H_' + str(m): home_places[m] for m in range(len(home_places))}
        work_location_dict = {'W_' + str(m): work_places[m] for m in range(len(work_places))}
        other_location_dict = {'O_' + str(m): other_places[m] for m in range(len(other_places))}
        lp_location_dict = {}
        for d in [home_location_dict, work_location_dict, other_location_dict]:
            lp_location_dict.update(d)
        return lp_location_dict

    # 根据原始label了significant places的轨迹文件提取出significant place对应的经纬度
    def lp_file2_loc_dict_lonlat(self, current_lp_df):
        home_place_num = len(current_lp_df['home_label_order'].unique()) - 1
        work_place_num = len(current_lp_df['work_label_order'].unique()) - 1
        other_place_num = len(current_lp_df['other_label_order'].unique()) - 1
        home_places = []
        work_places = []
        other_places = []
        for k in range(home_place_num):
            home_places.append(
                current_lp_df[current_lp_df['home_label_order'] == k][['home_lat', 'home_lon']].iloc[0].values)
        for k in range(work_place_num):
            work_places.append(
                current_lp_df[current_lp_df['work_label_order'] == k][['work_lat', 'work_lon']].iloc[0].values)
        for k in range(other_place_num):
            other_places.append(
                current_lp_df[current_lp_df['other_label_order'] == k][['other_lat', 'other_lon']].iloc[0].values)
        # 2.建立统一结构的lp_code-location字典
        home_location_dict = {'H_' + str(m): home_places[m] for m in range(len(home_places))}
        work_location_dict = {'W_' + str(m): work_places[m] for m in range(len(work_places))}
        other_location_dict = {'O_' + str(m): other_places[m] for m in range(len(other_places))}
        lp_location_dict = {}
        for d in [home_location_dict, work_location_dict, other_location_dict]:
            lp_location_dict.update(d)
        return lp_location_dict

    # LP LIST WITH LOC INFO
    def lp_list_with_loc_info(self,
                              OUT_PUT_FOLDER_INDIVIDUAL,
                              OUT_PUT_FOLDER,
                              user_num=2,
                              ):
        lp_file_list = []
        minitools.get_file_path(self.ori_lp_folder, lp_file_list, dir_list=[], target_ext='.csv')

        total_lp_code_list = []
        total_key_point_dict = {}
        for i in tqdm(range(user_num)):
            # temp_total_lp_code_df = total_lp_code_df.copy()
            # # 1.Get the hour state
            current_lp_df = pd.read_csv(lp_file_list[i])
            # hour_state_list = []
            # start_flag = False
            # for j in range(len(current_lp_df)):
            #     hour = current_lp_df.loc[j, 'hour']
            #     endhour = current_lp_df.loc[j, 'endhour']
            #     all_detect_label = current_lp_df.loc[j, 'all_detect_label']
            #     home_label = current_lp_df.loc[j, 'home_label_order']
            #     work_label = current_lp_df.loc[j, 'work_label_order']
            #     other_label = current_lp_df.loc[j, 'other_label_order']
            #     if all_detect_label > -1:
            #         start_flag = True
            #     if start_flag == False:
            #         continue
            #     if home_label > -1:
            #         for j in range(hour, endhour):
            #             hour_state_list.append({j: 'H_' + str(home_label)})
            #     elif work_label > -1:
            #         for j in range(hour, endhour):
            #             hour_state_list.append({j: 'W_' + str(work_label)})
            #     elif other_label > -1:
            #         for j in range(hour, endhour):
            #             hour_state_list.append({j: 'O_' + str(other_label)})
            # # 2. From hour state to lp code
            # for k in range(len(hour_state_list) - 1):
            #     (hour, state), = hour_state_list[k].items()
            #     (next_hour, next_state), = hour_state_list[k+1].items()
            #     temp_total_lp_code_df.loc[next_hour, state + '.' + next_state] = temp_total_lp_code_df.loc[next_hour, state + '.' + next_state] + 1

            # 3. Add the location information
            # temp_lp_location_dict = lpFile2LocDict_Meshcode(current_lp_df)
            temp_lp_location_dict = self.lp_file2_loc_dict_lonlat(current_lp_df)
            total_key_point_dict.update({lp_file_list[i].split('\\')[-1].split('_')[0]: temp_lp_location_dict})
            # #生成一个columns*2的df再append到lp_code_df 中
            # loc_info_O_list = []
            # loc_info_D_list = []
            # for column in temp_total_lp_code_df.columns:
            #     try:
            #         loc_info_O_list.append(temp_lp_location_dict[column.split('.')[0]])
            #     except:
            #         loc_info_O_list.append(0)
            #     try:
            #         loc_info_D_list.append(temp_lp_location_dict[column.split('.')[1]])
            #     except:
            #         loc_info_D_list.append(0)
            # temp_total_lp_code_df = temp_total_lp_code_df.append(DataFrame(columns=temp_total_lp_code_df.columns, data=[loc_info_O_list]))
            # temp_total_lp_code_df = temp_total_lp_code_df.append(DataFrame(columns=temp_total_lp_code_df.columns, data=[loc_info_D_list]))
            #
            #
            # #4. append lp matrix
            # total_lp_code_list.append(temp_total_lp_code_df.values)

        minitools.save_pkl(total_key_point_dict, OUT_PUT_FOLDER_INDIVIDUAL + str(user_num) + '_key_point_lonlat.pkl')
        DataFrame(total_key_point_dict).T.to_csv(OUT_PUT_FOLDER_INDIVIDUAL + str(user_num) + '_key_point_lonlat.csv')
        minitools.save_pkl(total_lp_code_list, OUT_PUT_FOLDER + str(user_num) + '_input_lp_loc_list.pkl')
        return

