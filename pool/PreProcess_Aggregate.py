# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for data aggregation or filtering'
__author__ = 'Li Peiran'

# 3.Import the modules.
import sys
import numpy as np
from pandas import DataFrame
from scipy import sparse
from tqdm import tqdm

# Load the function file in another file
sys.path.append('./')


# 4.Define the global variables. (if exists)

# 5.Define the class (if exsists)

# 6.Define the function (if exsists)


def aggregateHour(self, load_exsisting_file_flag=0):
    # resample the gounrd-truth S_matrix and SA_GT_df
    def resampleHour(input_hour):
        if (input_hour >= 700 and input_hour < 1200):
            return 1  # 'moring'
        elif (input_hour >= 1200 and input_hour < 1700):
            return 2  # 'afternoon'
        elif (input_hour >= 1700 and input_hour < 2300):
            return 3  # 'evening'
        else:
            return 4  # 'night'

    SA_true_df = self.SA_GT_df
    SA_true_df['Date'] = [x[0] for x in SA_true_df.index.values]
    SA_true_df['Hour'] = [x[1] for x in SA_true_df.index.values]
    SA_true_df['Meshcode'] = [x[2] for x in SA_true_df.index.values]
    SA_true_df['Hour_Resample'] = [resampleHour(x) for x in SA_true_df['Hour'].values]
    SA_true_df['Key_resample'] = [(x, y, z) for x, y, z in
                                  SA_true_df[['Date', 'Hour_Resample', 'Meshcode']].values]
    SA_true_df.index = SA_true_df['Key_resample'].values

    self.SA_GT_df = SA_true_df.groupby('Key_resample').mean()

    # resample the input st data
    # Aggregate the hour into 4 part for input id_st
    if (load_exsisting_file_flag != 1):
        id_st_list_sparse_csr = self.S_matrix  # id_st_dict_sparse
        id_st_list_resample_sparse_csr = []
        for i in tqdm(range(id_st_list_sparse_csr.shape[0])):
            temp_df = DataFrame(np.array(id_st_list_sparse_csr[i].todense())[0])
            temp_df['Key_resample'] = SA_true_df.index.values
            temp_df = temp_df.groupby('Key_resample').sum()
            temp_values = temp_df.values
            temp_values = np.where(temp_values >= 1, 1, 0)
            id_st_list_resample_sparse_csr.append(temp_values)
        # id_st_list_resample_sparse_csr = np.array(id_st_list_resample_sparse_csr)[:,:,0]
        id_st_list_resample_sparse_csr = sparse.csr_matrix(id_st_list_resample_sparse_csr)
        id_st_zip = zip(self.xyz_df.index.values, id_st_list_resample_sparse_csr)
        id_st_dict = dict(id_st_zip)
        self.S_matrix = id_st_list_resample_sparse_csr
        self.S_matrix_dict = id_st_dict


def selectPartID(self, sampling_rate=0.5):
    # 1. define the selected ids
    ids_for_train = DataFrame(self.S_matrix_dict.keys()).sample(frac=sampling_rate, random_state=3,
                                                                replace=False).index.values  # False:取行数据后不放回，下次取其它行数据
    ids_dict_for_train = \
        DataFrame(self.S_matrix_dict.keys()).sample(frac=sampling_rate, random_state=3, replace=False)[0].values
    # 2. 筛出S里面train里面相应的部分
    S_filter_matrix = self.S_matrix[ids_for_train]
    id_S_zip_filter = zip(ids_dict_for_train, S_filter_matrix)
    S_filter_matrix_dict = dict(id_S_zip_filter)
    # 3. 增加类中与train有关的变量
    self.xyz_train_df = self.xyz_df.loc[ids_dict_for_train]
    # 4. 筛出SA中总人数为0的部分
    # 4.1. 筛出ground-truth里面key里面经过数>0的部分
    SA_train_df = self.SA_GT_df.copy()
    SA_train_df['index_for_filter'] = range(len(SA_train_df))
    SA_train_df['with_pass'] = np.array(np.sum(S_filter_matrix, axis=0))[0]
    filter_0 = SA_train_df[SA_train_df['with_pass'] > 0].index_for_filter.values
    SA_train_filter_df = SA_train_df[SA_train_df['with_pass'] > 0]
    SA_train_filter_df = SA_train_filter_df.drop(['with_pass', 'index_for_filter'], axis=1)
    # SA_train_filter_df = SA_train_filter_df.drop(['Hour', 'Meshcode', 'Hour_Resample'], axis=1)
    self.SA_train_df = SA_train_filter_df

    # 4.2. 筛出id_st里面key里面相应的部分
    S_filter_matrix = S_filter_matrix.T[filter_0, :].T
    id_S_zip_filter = zip(self.xyz_train_df.index.values, S_filter_matrix)
    S_filter_matrix_dict = dict(id_S_zip_filter)
    self.S_train_matrix = S_filter_matrix
    self.S_train_matrix_dict = S_filter_matrix_dict


# if __name__ == '__main__':
