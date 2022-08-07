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
from skmob.measures.individual import home_location
import multiprocessing as mp
import threading as td
import time
import pickle
import jismesh.utils as ju
import pandas as pd
from pandas import DataFrame
from scipy import sparse
from utils import MiniTools
import collections
# Load the function file in another file
#sys.path.append('../')
import MobakuProcessing


# 4.Define the global variables. (if exists)

# 5.Define the class (if exsists)

# 6.Define the function (if exsists)


def segmentedTraj2HourMesh(INPUT_FILE_PATH, OUT_PUT_FOLDER_PATH, filter_period=[]):
    try:
        # Read the original file
        traj_df = pd.read_csv(INPUT_FILE_PATH, header=None)
        traj_df.columns = ['uid', 'date', 'Unnamed: 2', 'transportation_mode', 'move_stay',
                           'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9',
                           'start_time', 'end_time', 'Unnamed: 12', 'Unnamed: 13',
                           'Unnamed: 14', 'Unnamed: 15', 'trajectory']
        # Exclude useless columns
        traj_df = traj_df[['uid', 'transportation_mode', 'start_time', 'end_time', 'trajectory']]


        # 拆出每个timestamp的轨迹经纬度
        # pat =分割字符或者正则表达式，n=拆分段数，-1为默认，expand=ture直接拆成dataframe
        traj_list_df = traj_df['trajectory'].str.split(pat=';', n=-1, expand=True)
        # 逐点重塑dataframe（先建立dict，然后DataFrame）
        dict_list = []
        for i in traj_list_df.index:
            # print(i,'/', len(traj_list_df.index))
            for j in traj_list_df.columns:
                temp_str = traj_list_df.iloc[i, j]
                if (temp_str != None):
                    # print(temp_str)
                    temp_str_list = temp_str.split('|')
                    temp_dict = {'uid': traj_df['uid'][i],
                                 'start_str_time': traj_df['start_time'][i],
                                 'end_str_time': traj_df['end_time'][i],
                                 'segment_id': i,
                                 'transportation_mode': traj_df['transportation_mode'][i],
                                 'point_id': temp_str_list[0],
                                 'point_str_time': temp_str_list[1],
                                 'lat': float(temp_str_list[2]),
                                 'lon': float(temp_str_list[3])}
                    dict_list.append(temp_dict)
        traj_df = DataFrame(dict_list)
        # #Exclude error record
        traj_df = traj_df[traj_df['lat'] != 0]
        traj_df = traj_df[traj_df['lon'] != 0].reset_index()

        # 1. For STAY
        traj_stay_df = traj_df[traj_df['transportation_mode'] == 'STAY'].copy()
        # Get the time_stamp
        traj_stay_df['start_struct_time'] = [time.strptime(x, "%Y-%m-%d %H:%M:%S") for x in
                                             traj_stay_df['start_str_time'].values]
        traj_stay_df['start_time_stamp'] = [time.mktime(x) for x in traj_stay_df['start_struct_time'].values]
        traj_stay_df['end_struct_time'] = [time.strptime(x, "%Y-%m-%d %H:%M:%S") for x in
                                           traj_stay_df['end_str_time'].values]
        traj_stay_df['end_time_stamp'] = [time.mktime(x) for x in traj_stay_df['end_struct_time'].values]
        # Filter the points not in assigned period
        if(filter_period != []):
            traj_stay_df = traj_stay_df[(traj_stay_df['start_time_stamp'] > filter_period[0])
                                        & (traj_stay_df['start_time_stamp'] < filter_period[1])]
            filter_stay_remain = len(traj_stay_df)
        else:
            pass
        # Initial a new list for dataframe generation
        stay_hour_location_df_list = []
        # 从dataframe切出numpy values
        start_time_stamp_np = traj_stay_df['start_time_stamp'].values
        end_time_stamp_np = traj_stay_df['end_time_stamp'].values
        uid_np = traj_stay_df['uid'].values
        segment_id_np = traj_stay_df['segment_id'].values
        transportation_mode_np = traj_stay_df['transportation_mode'].values
        lat_np = traj_stay_df['lat'].values
        lon_np = traj_stay_df['lon'].values

        for i in range(len(traj_stay_df)):
            # for every segment_id
            # 这一步都按绝对时间考虑，考虑相对时间在后面读入mobaku后
            # 中间过程以时间戳为基准，以后根据时间戳更新struct_time
            # 1.Initial variable in this segmentation 节省时间，不然下面每循环都要读一次df
            start_time_stamp_i = start_time_stamp_np[i]
            end_time_stamp_i = end_time_stamp_np[i]
            uid_i = uid_np[i]
            segment_id_i = segment_id_np[i]
            transportation_mode_i = transportation_mode_np[i]
            lat_i = lat_np[i]
            lon_i = lon_np[i]

            current_time_stamp_i = start_time_stamp_i
            while (current_time_stamp_i < end_time_stamp_i + 3600):
                temp_dict = {'uid': uid_i,
                             'segment_id': segment_id_i,
                             'transportation_mode': transportation_mode_i,
                             'point_time_stamp': current_time_stamp_i,
                             'lat': lat_i,
                             'lon': lon_i}
                stay_hour_location_df_list.append(temp_dict)
                current_time_stamp_i = current_time_stamp_i + 3600
        traj_stay_df = DataFrame(stay_hour_location_df_list)

        # 2. For MOVE
        traj_move_df = traj_df[traj_df['transportation_mode'] != 'STAY'].copy()
        # Get the time stamp
        traj_move_df['point_struct_time'] = [time.strptime(x, "%Y-%m-%d %H:%M:%S") for x in
                                             traj_move_df['point_str_time'].values]
        traj_move_df['point_time_stamp'] = [time.mktime(x) for x in traj_move_df['point_struct_time'].values]
        #Filter the points not in assigned period
        if (filter_period != []):
            traj_move_df = traj_move_df[(traj_move_df['point_time_stamp'] > filter_period[0])
                                        & (traj_move_df['point_time_stamp'] <filter_period[1])]
            filter_move_remain = len(traj_move_df)
        else:
            pass
        if filter_stay_remain+filter_move_remain ==0:
            print('File (%s) has not records in filter perios. \n' % INPUT_FILE_PATH.split('\\')[-1])
            return 0
        #Drop useless information
        traj_move_df = traj_move_df[['uid', 'segment_id', 'transportation_mode', 'point_time_stamp', 'lat', 'lon']]
        # 3. Merge traj_move and traj_stay
        traj_df = pd.concat([traj_move_df, traj_stay_df], axis=0)
        traj_df.sort_values(by=['point_time_stamp'], axis=0, ascending=['True'], inplace=True)
        # get the year+month+day,day_of_week,hour,meshcode
        traj_df['point_struct_time'] = [time.localtime(x) for x in traj_df['point_time_stamp'].values]
        traj_df['year_month_day'] = ['%04d%02d%02d' % (x.tm_year, x.tm_mon, x.tm_mday) for x in
                                     traj_df['point_struct_time'].values]
        traj_df['day_of_week'] = [x.tm_wday for x in traj_df['point_struct_time'].values]
        traj_df['hour'] = [x.tm_hour for x in traj_df['point_struct_time'].values]
        traj_df['meshcode'] = ju.to_meshcode(traj_df['lat'].values, traj_df['lon'].values, 4)
        # Drop duplicates
        traj_df.drop_duplicates(subset=['year_month_day', 'hour', 'meshcode'], keep='first', inplace=True)
        # Drop useless columns
        traj_df.reset_index(inplace=True)  # for better look, no physical meaning
        traj_df.drop(['index', 'segment_id', 'point_struct_time'], axis=1, inplace=True)

        # If used in Linux system, change '\\' to '/'
        traj_df.to_csv(OUT_PUT_FOLDER_PATH + INPUT_FILE_PATH.split('\\')[-1])

        return 1

    except Exception as ex:
        print('File (%s) Error occurs: %s \n' % (INPUT_FILE_PATH.split('\\')[-1],ex))
        pass


def mpSegmentedTraj2HourMesh(traj_file_list, OUT_PUT_FOLDER_PATH, multiprocessing_core=4, filter_period=[]):
    if multiprocessing_core > 1:
        pool = mp.Pool(processes=multiprocessing_core)
        # 这里使用imap而不是apply_async是为了正常使用tqdm进度条,但是imap不能传入多个参数，失败，只能放弃使用进度条
        input_para_list = [(x, OUT_PUT_FOLDER_PATH, filter_period) for x in traj_file_list]
        pool.starmap(segmentedTraj2HourMesh, input_para_list)
        pool.close()
        pool.join()
        print('Mission Completed.')
    else:
        for i in tqdm(range(len(traj_file_list))):
            segmentedTraj2HourMesh(traj_file_list[i], OUT_PUT_FOLDER_PATH, filter_period)


def generateVectorSfromHourMeshTraj(mobaku_hour_mesh_str_key, INPUT_HOUR_MESH_TRAJ_FILE_PATH,
                                    matching_mode='by_day_of_week'):
    """
    Generate s-vector from mobaku-s-vector and hour-mesh-traj file
    Note:
    The format of Mobaku is:
        hour: 0,100,200,300...,2300
        day_of_week:11,12,...17
    The format of time.struct_time is:
        hour:0,1,2,...,23
        day_of_week:0,1,2,3...,6
    To match the format of Mobaku,here, we trans the struct_time to match the Mobaku.

    Args:
        Mobaku data folder path
        requires:
        1. mobaku_hour_mesh (contains all key list):
        from function getSAfromMobakuPkl(),which in the PreProcess_Mobaku.py
        then trans demographic_df in string format
        2. INPUT_HOUR_MESH_TRAJ_FILE_PATH:
        generated by function multiPoolSegmentedTraj2HourMesh()
        3. matching_mode:
        use 'by_absolute_date' or 'by_day_of_week' to match the time
    Retruns:
        s-vector for a certain user
    """

    # Get the hour mesh data
    temp_traj_df = pd.read_csv(INPUT_HOUR_MESH_TRAJ_FILE_PATH, index_col=0)
    # Generate the Key
    if matching_mode == 'by_absolute_date':
        temp_traj_df['key'] = temp_traj_df.apply(lambda row: (row.year_month_day, row.hour * 100, row.meshcode), axis=1)
    elif matching_mode == 'by_day_of_week':
        temp_traj_df['key'] = temp_traj_df.apply(lambda row: (row.day_of_week + 11, row.hour * 100, row.meshcode),
                                                 axis=1)
    else:
        print('Please give corret matching mode.')
        return 0

    # Generate s vector of a certain user
    # WARNING!!!!:np.isin函数不能判断两个元组/数组是否相等，这里它会把数组展开为元素进行判断，
    # 所以，必须要转换成string的形式，不然结果是错误的
    traj_hour_mesh = np.array([str(x[0]) + str(x[1]) + str(x[2]) for x in temp_traj_df.key.values])
    try:
        s_vector = np.isin(mobaku_hour_mesh_str_key, traj_hour_mesh).astype(np.uint8)
        #Considering the count
        key_count = collections.Counter(traj_hour_mesh)
        key_count = {k: v for k, v in key_count.items() if v > 1} #to improve speed, deal count above 1 only
        for key in key_count.keys():
            s_vector[np.where(s_vector == key)] = key_count[key]
    except Exception as ex:
        print('File (%s) Error occurs: %s' % (INPUT_HOUR_MESH_TRAJ_FILE_PATH.split('\\')[-1],ex))
        print('Thus,a zero numpy will be return for %s' % INPUT_HOUR_MESH_TRAJ_FILE_PATH.split('\\')[-1])
        return np.zeros(len(traj_hour_mesh))

    return s_vector


def generateMatrixSfromHourMeshTraj(mobaku_hour_mesh_str_key, INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST, user_ids_list,
                                    SAVE_FILE_PATH, matching_mode='by_day_of_week', computing_mode='add'):
    """
    Generate S-Matrix from mobaku-s-vector and hour-mesh-traj file,
    based on function makeVectorSfromHourMeshTraj

    Args:
        1. mobaku_hour_mesh <str list>: from function getSAfromMobakuPkl(),which in the PreProcess_Mobaku.py then trans demographic_df in string format
        2. INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST<str>: generated by function multiPoolSegmentedTraj2HourMesh(),here it should be the file list of users who you want to take in account
        3. user_ids_list<list>: user ids who want to take in account (corresponding to the files)
        4. SAVE_FILE_PATH (xxx.npz)<str>: Save the S-Matrix dict to avoid computing it every time even if same users
        5. matching_mode <str>: use 'by_absolute_date' or 'by_day_of_week' to match the time
        6. computing_mode <str>: 'cover': cover existed files or 'add': skip existed files
    Retruns:
        S-Matrix for a set of users
    """
    # If save_folder_path does not exist, create one
    MiniTools.ifFolderExistThenCreate(SAVE_FILE_PATH)
    # Get the age/gender information from hour stay data
    s_vector_list = []
    s_vector_path_list = []
    if computing_mode == 'cover':
        for i in tqdm(range(len(INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST))):
            temp_s_vector = generateVectorSfromHourMeshTraj(mobaku_hour_mesh_str_key,
                                                            INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST[i], matching_mode)
            uid = os.path.basename(INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST[i]).split('.')[0]
            temp_save_path = os.path.join(SAVE_FILE_PATH, uid + '.npz')
            sparse.save_npz(temp_save_path, sparse.csr_matrix(temp_s_vector))

    elif computing_mode == 'add':
        #Get the current vectorS file list
        current_vector_list = []
        MiniTools.getFilePath(SAVE_FILE_PATH,file_list=current_vector_list,target_ext='.npz')
        current_vector_list = [os.path.basename(x) for x in current_vector_list]
        for i in tqdm(range(len(INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST))):
            uid = os.path.basename(INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST[i]).split('.')[0]
            temp_save_path = os.path.join(SAVE_FILE_PATH, uid + '.npz')
            # If the file not exists, then do:
            if os.path.basename(temp_save_path) not in current_vector_list:
                temp_s_vector = generateVectorSfromHourMeshTraj(mobaku_hour_mesh_str_key,
                                                                INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST[i], matching_mode)
                sparse.save_npz(temp_save_path, sparse.csr_matrix(temp_s_vector))


    for i in tqdm(range(len(INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST))):
        uid = os.path.basename(INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST[i]).split('.')[0]
        temp_save_path = os.path.join(SAVE_FILE_PATH, uid + '.npz')
        temp_s_vector = sparse.load_npz(temp_save_path)
        s_vector_list.append(temp_s_vector)
        s_vector_path_list.append(temp_s_vector)

    # s_vector_zip = zip(user_ids_list, s_vector_list)
    # s_vector_dict = dict(s_vector_zip)

    # Save the sparse-type S-matrix and .npy-type
    # id_st_list_sparse = sparse.csr_matrix(s_vector_list)
    # sparse.save_npz(SAVE_FILE_PATH+'.npz', id_st_list_sparse)
    S_matrix = sparse.vstack(s_vector_list)
    sparse.save_npz(os.path.join(SAVE_FILE_PATH, 'S-matrix.npz'),S_matrix)
    np.save(os.path.join(SAVE_FILE_PATH, 'user_ids_list.npy'), user_ids_list)

    print('Matrix-S computation Completed, saved in %s' % SAVE_FILE_PATH)
    return 1


if __name__ == '__main__':
    # 1. Get all of the trajectory file path
    #TRAJ_PATH = r'F:\TrajData\2013\2013_traj'
    TRAJ_PATH = r'..\assets\Traj'
    folder_list = []
    total_traj_file_list = []
    MiniTools.getFilePath(TRAJ_PATH, total_traj_file_list, folder_list, '.csv')
    total_ids = [x.split('\\')[-1].split('.')[0] for x in total_traj_file_list]

    # 2. The life-pattern people id (those who are need to be labeled)should be given
    # to find corresponding file from all traj files
    #NEED_LABELED_IDS_PATH = r'F:\life_pattern_detect\great_tokyo\6_NMF_result\filename_multiple_HW_single_O_total_day.csv'
    NEED_LABELED_IDS_PATH = r'..\assets\Traj\filename_multiple_HW_single_O_total_day.csv'
    xyz_ids_df = pd.read_csv(NEED_LABELED_IDS_PATH)
    xyz_ids_df.columns = ['ID']
    xyz_ids = ['%08d' % x for x in xyz_ids_df['ID'].values]
    all_xyz_traj_file_list = np.array(total_traj_file_list)[np.isin(total_ids, xyz_ids)]

    # 3. Preprocess the traj data for S-matrix generation (the next step)
    #PROCESSED_TRAJ_PATH = r'F:\TrajData\2013\2013_traj_hour_meshcode_20130601~20130701/'
    PROCESSED_TRAJ_PATH = r'..\assets\Processed_Traj\\'#It's only for result save-path, you can assign it win anywhere
    MiniTools.ifFolderExistThenCreate(PROCESSED_TRAJ_PATH)
    #calcute the input filter time stamp
    start_str_time = "2013-06-01 00:00:00"
    end_str_time = "2013-06-14 00:00:00"
    filter_period_start_stamp = time.mktime(time.strptime(start_str_time, "%Y-%m-%d %H:%M:%S"))
    filter_period_end_stamp = time.mktime(time.strptime(end_str_time, "%Y-%m-%d %H:%M:%S"))
    filter_period = [filter_period_start_stamp,filter_period_end_stamp]
    #Call the function
    mpSegmentedTraj2HourMesh(all_xyz_traj_file_list, PROCESSED_TRAJ_PATH, multiprocessing_core = 3, filter_period=filter_period)


    # 4. Get the str key
    MOBAKU_FOLDER_PATH = r'..\assets\00_cencus_data\\'
    MOBAKU_PKL_FOLDER_PATH = r'..\assets\mobaku_demographic_pkl/'
    MOBAKU_DEMOGRAPHIC_PATH = MOBAKU_FOLDER_PATH + '02_性年代(10歳階).csv'
    # transMobaku2Shape(MOBAKU_FOLDER_PATH,OUTPUT_FOLDER_PATH)
    # transMobaku2PKL(MOBAKU_DEMOGRAPHIC_PATH, MOBAKU_PKL_FOLDER_PATH)
    demographic_df = MobakuProcessing.getSAfromMobakuPkl(MOBAKU_PKL_FOLDER_PATH, load_mode='by_day_of_week')
    str_key = np.array([str(x[0]) + str(x[1]) + str(x[2]) for x in demographic_df.index.values])

    # 5. Generate the Matrix S
    need_labeled_traj_file_afterprocessing_list = []
    MiniTools.getFilePath(PROCESSED_TRAJ_PATH, need_labeled_traj_file_afterprocessing_list, dir_list=[], target_ext='.csv')
    xyz_ids = [os.path.basename(x).split('.')[0] for x in need_labeled_traj_file_afterprocessing_list]
    #S_Matrix_SAVE_FILE_PATH = '../assets/S-matrix-20130601_20130701/'
    S_Matrix_SAVE_FILE_PATH = '../assets/S-matrix-test/'
    MiniTools.ifFolderExistThenCreate(S_Matrix_SAVE_FILE_PATH)
    generateMatrixSfromHourMeshTraj(mobaku_hour_mesh_str_key=str_key,
                                    INPUT_HOUR_MESH_TRAJ_FILE_PATH_LIST=need_labeled_traj_file_afterprocessing_list,
                                    user_ids_list=xyz_ids,
                                    SAVE_FILE_PATH=S_Matrix_SAVE_FILE_PATH,
                                    matching_mode='by_day_of_week',
                                    computing_mode='add')
