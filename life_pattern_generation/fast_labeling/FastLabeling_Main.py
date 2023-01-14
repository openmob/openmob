# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-


# 2.Note for this file.
'This file is for fast-lebeling model training'
__author__ = 'Li Peiran'

import datetime

import numpy as np
import pandas as pd
import scipy.stats
# For NN
import torch
import torch.nn as nn
from pandas import DataFrame
from pyswarm import pso
from scipy import sparse
from torch.autograd import Variable
# 3.Import the modules.
from tqdm import tqdm

# Load the function file in another file
# import MobakuProcessing
from utils import MiniTools
# from models.FNN import Activation_Net
# from models.MultiTaskFNN import Activation_Multi_Net


# 4.Define the global variables. (if exists)

# 5.Define the class (if exsists)

# 6.Define the function (if exsists)
def predictEval(group_number,optimization_parameter,x, y, z):
    m = group_number
    parameter = optimization_parameter
    Mean_Dynamic_x = parameter[0:m]  # 注意，左边闭集，右边开集！
    Mean_Dynamic_y = parameter[m:2 * m]
    Mean_Dynamic_z = parameter[2 * m:3 * m]
    Std_Dynamic = parameter[3 * m:4 * m]

    # Renew the state of every uid
    A = np.zeros([group_number,len(x)])
    for i in range(m):
        temp_x_square = np.power((x - Mean_Dynamic_x[i]), 2)
        temp_y_square = np.power((y - Mean_Dynamic_y[i]), 2)
        temp_z_square = np.power((z - Mean_Dynamic_z[i]), 2)
        temp_distance_list = np.sqrt((temp_x_square + temp_y_square + temp_z_square))
        temp_A = MiniTools.getGaussianPob(0, Std_Dynamic[i], temp_distance_list)  # for spatial_distance in np.array(temp_distance_list)]
        A[i] = temp_A
    return MiniTools.normSumAxis1(A.T)


class FastLabeling():
    '''
    This is for test.
    '''
    def __call__(self):
        return 1

    def __init__(self, MOBAKU_SA_GT_PATH, LIFE_PATTERN_XYZ_PATH, S_MATRIX_FILE_PATH, INPUT_USER_LIST_PATH, FL_RESULT_SAVE_PATH=r'F:\id_st_list\Test_Result\temp'):
        '''
        This is for initial class.
        '''

        # 1. Load the SA_GT_df (age and gender ground-truth data)
        self.SA_GT_df = pd.read_csv(MOBAKU_SA_GT_PATH,index_col=0)
        # 2. Load the S (sparse matrix storing users' spatio-temporal state)
        self.S_matrix = sparse.load_npz(S_MATRIX_FILE_PATH)
        # 3. Load the user_ids_list.npy (which is coresponding to S_Matrix)
        self.users_list = np.load(INPUT_USER_LIST_PATH).astype(int)
        # 4. Get the total life-pattern xyz coordinate file
        xyz_df = pd.read_csv(LIFE_PATTERN_XYZ_PATH,index_col=0)
        xyz_df.columns = ['X', 'Y', 'Z']
        xyz_df = xyz_df.loc[self.users_list]
        self.xyz_df = xyz_df
        # Get the used xyz files from total xyz coordinate file
        #xyz_df = xyz_df[xyz_df.index in input_users_list]
        # 5.Other settings
        self.group_number = 16 # Default = 16
        self.group_label = ['0-15*M', '0-15*F', '15-20*M', '15-20*F', '20-30*M', '20-30*F', '30-40*M', '30-40*F',
                            '40-50*M', '40-50*F', '50-60*M', '50-60*F', '60-70*M', '60-70*F', '70-80*M', '70-80*F']
        self.FL_RESULT_SAVE_PATH = FL_RESULT_SAVE_PATH


    # def savestate2File(self, obj_name):
    #     # Save the object parameters
    #     Prameter_dict = {'group_number': self.group_number,
    #                      'parameter': self.optimization_parameter,
    #                      'mean_x': self.optimization_parameter[0:self.group_number],
    #                      'mean_y': self.optimization_parameter[self.group_number:self.group_number * 2],
    #                      'mean_z': self.optimization_parameter[self.group_number * 2:self.group_number * 3],
    #                      'std': self.optimization_parameter[self.group_number * 3:self.group_number * 4],
    #                      'uid_num': self.uid_num,
    #                      'group_label': self.group_label}
    #     print(1)
    #     savePKL(Prameter_dict, obj_name + 'parameter.pkl')
    #
    #     # Save the current data
    #     self.xyz_df.to_csv(obj_name + 'input_xyz_data.csv')
    #     self.uid_state.to_csv(obj_name + 'uid_pob_state.csv')

    def groupFilter(self):
        '''
        This is for demographic group filtering.
        '''
        # 3. 去掉<15及>70岁的ground-truth and label
        self.group_number = 16 - 4
        self.SA_GT_df = self.SA_GT_df[self.SA_GT_df.columns[2:-2]]
        self.group_label = self.group_label[2:-2]

    def filterKeyByNumber(self, threshold_number=10):
        if (threshold_number > 0):
            # 1. Initialization
            SA_GT_df = self.SA_GT_df
            S_matrix = self.S_matrix
            # 2. Filter users number > threshold_number
            SA_GT_df['index_for_filter'] = range(len(SA_GT_df))
            SA_GT_df['with_pass'] = np.array(np.sum(S_matrix, axis=0))[0]
            SA_filter_df = SA_GT_df[SA_GT_df['with_pass'] > threshold_number]
            SA_filter_df = SA_filter_df.drop(['with_pass', 'index_for_filter'], axis=1)
            # 3. Filter the S-matrix to match the SA_GT's keys
            filter_index = SA_GT_df[SA_GT_df['with_pass'] > threshold_number].index_for_filter.values
            S_filter_matrix = S_matrix.T[filter_index, :].T
            #4. Update the S-matrix and SA_GT
            self.SA_GT_df = SA_filter_df
            self.S_matrix = S_filter_matrix
            return 1
        else:
            return 0

    def filterUserByNumber(self, threshold_number=10):
        if (threshold_number > 0):
            # 1. Initialization
            user_list = self.users_list
            xyz_df = self.xyz_df
            S_matrix = self.S_matrix
            # 2. Filter users number > threshold_number
            xyz_df['index_for_filter'] = range(len(xyz_df))
            xyz_df['pass_key_number'] = np.array(np.sum(S_matrix, axis=1))[:,0]
            filter_index = xyz_df[xyz_df['pass_key_number'] > threshold_number].index_for_filter.values
            # 3. Filter the S-matrix to match the SA_GT's keys
            S_filter_matrix = S_matrix[filter_index]
            #4. Update the S-matrix and SA_GT
            xyz_df = xyz_df[xyz_df['pass_key_number']>threshold_number]
            xyz_df = xyz_df.drop(['pass_key_number', 'index_for_filter'], axis=1)
            self.xyz_df = xyz_df
            self.users_list = xyz_df.index.values
            self.S_matrix = S_filter_matrix
            return 1
        else:
            return 0

    # Aggregate the hour into 4 part for ground-truth
    def saveGaussianCurrentResult(self, A_numpy, SA_df, loss, save_epoch=30):
        # 保存结果
        print('Current Loss: ', loss)
        self.train_processing.append({'epoch': self.current_iter_num, 'loss': loss})

        if (self.current_iter_num == 0):
            # 创建文件夹
            MiniTools.ifFolderExistThenCreate(self.FL_RESULT_SAVE_PATH)
            # 存一次ground-truth data，后面的循环不再存
            self.SA_GT_df.to_csv(self.FL_RESULT_SAVE_PATH + '/SA_GT.csv')
            # 存一次归一化后的ground-truth data，后面的循环不再存
            groundtruth_st_data_norm = DataFrame(MiniTools.normSumAxis1(self.SA_GT_df.values), columns=self.SA_GT_df.columns)
            groundtruth_st_data_norm.to_csv(self.FL_RESULT_SAVE_PATH + '/SA_GT_norm.csv')
            # 存一次xyz coordinate，后面的循环不再存
            self.xyz_df.to_csv(self.FL_RESULT_SAVE_PATH + '/xyz.csv')
        if (self.current_iter_num % save_epoch == 0):
            # 更新当前类中的每个uid状态值
            uid_state_df = DataFrame(A_numpy).T
            self.uid_state = uid_state_df
            self.uid_state.index = self.xyz_df.index
            # 输出当前类中的每个uid状态值（norm前的）
            self.uid_state.to_csv(self.FL_RESULT_SAVE_PATH + '/A_iter_%d.csv' % self.current_iter_num)
            # 3. 输出当前类中的每个uid状态值（norm后的）
            DataFrame(MiniTools.normSumAxis1(self.uid_state.values), index=self.xyz_df.index).to_csv(self.FL_RESULT_SAVE_PATH + '/A_iter_%d.csv' % self.current_iter_num)
            # 4. 输出当前类中的每个st状态值（norm前的）
            SA_df.to_csv(self.FL_RESULT_SAVE_PATH + '/SA_iter_%d.csv' % self.current_iter_num)
            # 5. 输出当前类中的每个st状态值（norm后的）
            st_ag_pos_state_df_norm = DataFrame(MiniTools.normSumAxis1(SA_df.values), columns=SA_df.columns)
            st_ag_pos_state_df_norm.to_csv(
                self.FL_RESULT_SAVE_PATH + '/SA_norm_iter_%d.csv' % self.current_iter_num)

    def saveNNCurrentResult(self, A_tensor, SA_tensor, loss, save_epoch=30, mode='Train'):
        # 保存结果
        print('Current Loss: ', loss)
        if (mode == 'Train'):
            self.train_processing.append({'epoch': self.current_iter_num, 'loss': loss.cpu().detach().numpy()})
        else:
            self.eval_processing.append({'epoch': self.current_iter_num, 'loss': loss.cpu().detach().numpy()})
        if (self.current_iter_num == 0):
            # 创建文件夹
            MiniTools.ifFolderExistThenCreate(self.FL_RESULT_SAVE_PATH + '/' + mode)
            # 存一次ground-truth data，后面的循环不再存
            self.SA_GT_df.to_csv(self.FL_RESULT_SAVE_PATH + '/' + mode + '/st_groundtruth.csv')
            # 存一次归一化后的ground-truth data，后面的循环不再存
            groundtruth_st_data_norm = DataFrame(MiniTools.normSumAxis1(self.SA_GT_df.values), columns=self.SA_GT_df.columns)
            groundtruth_st_data_norm.to_csv(self.FL_RESULT_SAVE_PATH + '/' + mode + '/norm_st_groundtruth.csv')

        if (self.current_iter_num % save_epoch == 0):
            # 更新当前类中的每个uid状态值
            # 1. 先将tensor转为numpy
            possibility_of_group = A_tensor.cpu().detach().numpy()
            st_ag_pos_state = SA_tensor.cpu().detach().numpy()
            st_ag_pos_state_df = DataFrame(st_ag_pos_state)
            # 2. 给当前类赋值每个uid状态值
            uid_state_df = DataFrame(possibility_of_group)
            self.uid_state = uid_state_df
            if (mode == 'Train'):
                self.uid_state.index = self.xyz_train_df.index
            else:
                self.uid_state.index = self.xyz_df.index
            # 输出当前类中的每个uid状态值（norm前的）
            self.uid_state.to_csv(self.FL_RESULT_SAVE_PATH + '/' + mode + '/uid_pob_state_iter%d.csv' % self.current_iter_num)
            # 3. 输出当前类中的每个uid状态值（norm后的）
            if (mode == 'Train'):
                DataFrame(MiniTools.normMinMaxAxis1(self.uid_state.values), index=self.xyz_train_df.index).to_csv(
                    self.FL_RESULT_SAVE_PATH + '/' + mode + '/uid_pob_state_norm_iter%d.csv' % self.current_iter_num)
            else:
                DataFrame(MiniTools.normMinMaxAxis1(self.uid_state.values), index=self.xyz_df.index).to_csv(
                    self.FL_RESULT_SAVE_PATH + '/' + mode + '/uid_pob_state_norm_iter%d.csv' % self.current_iter_num)
            # 4. 输出当前类中的每个st状态值（norm前的）
            st_ag_pos_state_df.to_csv(self.FL_RESULT_SAVE_PATH + '/' + mode + '/st_state_iter%d.csv' % self.current_iter_num)
            # 5. 输出当前类中的每个st状态值（norm后的）
            st_ag_pos_state_df_norm = DataFrame(MiniTools.normSumAxis1(st_ag_pos_state_df.values),
                                                columns=st_ag_pos_state_df.columns)
            st_ag_pos_state_df_norm.to_csv(
                self.FL_RESULT_SAVE_PATH + '/' + mode + '/st_state_norm_iter%d.csv' % self.current_iter_num)



    def calcuSAbyGaussianParameters(self, parameter,loss_type = 'MAE'):

        # Input the parameters
        m = self.group_number
        Mean_Dynamic_x = parameter[0:m]  # 注意，左边闭集，右边开集！
        Mean_Dynamic_y = parameter[m:m * 2]
        Mean_Dynamic_z = parameter[m * 2:m * 3]
        Std_Dynamic = parameter[m * 3:m * 4]
        #Group_scale = parameter[m * 4:m * 5]
        print('Current parameters:', parameter)

        # Renew the state of every uid
        A_list = []
        for i in range(m):
            temp_x_square = np.power((self.xyz_df.values[:, 0] - Mean_Dynamic_x[i]), 2)
            temp_y_square = np.power((self.xyz_df.values[:, 1] - Mean_Dynamic_y[i]), 2)
            temp_z_square = np.power((self.xyz_df.values[:, 2] - Mean_Dynamic_z[i]), 2)
            temp_distance_list = np.sqrt((temp_x_square + temp_y_square + temp_z_square))
            temp_pob_list = MiniTools.getGaussianPob(0, Std_Dynamic[i], np.array(
                temp_distance_list))  # for spatial_distance in np.array(temp_distance_list)]
            A_list.append(temp_pob_list)


        # Renew the state of ag_df
        SA_calcu = sparse.csr_matrix(np.array(A_list)).dot(self.S_matrix)
        SA_calcu_df = DataFrame(np.array(SA_calcu.todense()).T)


        # 对每个key计算统计值和ground-truth的差值
        if (loss_type == 'MAE'):
            loss = MiniTools.numpyMAE(MiniTools.normSumAxis1(self.SA_GT_df.values), MiniTools.normSumAxis1(SA_calcu_df.values))
        elif (loss_type == 'MSE'):
            loss = MiniTools.numpyMSE(MiniTools.normSumAxis1(self.SA_GT_df.values), MiniTools.normSumAxis1(SA_calcu_df.values))

        self.saveGaussianCurrentResult(A_list, SA_calcu_df, loss, save_epoch=30)
        self.current_iter_num = self.current_iter_num + 1
        #print(Mean_Dynamic_x,Mean_Dynamic_y,Mean_Dynamic_z,Std_Dynamic)
        print(loss)
        return loss

    def calcuSAbyGaussianParametersTensor(self, parameter_tensor,SA_GT_tensor,loss_type = 'MSE'):

        m = self.group_number
        Mean_Dynamic_x = parameter_tensor[0:m]  # 注意，左边闭集，右边开集！
        Mean_Dynamic_y = parameter_tensor[m:2 * m]
        Mean_Dynamic_z = parameter_tensor[2 * m:3 * m]
        Std_Dynamic = parameter_tensor[3 * m:4 * m]
        print(parameter_tensor)

        A_tensor = torch.FloatTensor(np.zeros((len(self.xyz_df), self.group_number)))
        for i in range(m):
            temp_x_square = torch.pow((self.xyz_torch_tensor[:, 0] - Mean_Dynamic_x[i]), 2)
            temp_y_square = torch.pow((self.xyz_torch_tensor[:, 1] - Mean_Dynamic_y[i]), 2)
            temp_z_square = torch.pow((self.xyz_torch_tensor[:, 2] - Mean_Dynamic_z[i]), 2)
            temp_distance_list = torch.sqrt((temp_x_square + temp_y_square + temp_z_square))
            temp_pob_list = MiniTools.getGaussianPobTensor(0, Std_Dynamic[i], temp_distance_list)
            # print(temp_pob_list)
            A_tensor[:, i] = temp_pob_list

        SA_calcu_tensor = torch.sparse.mm(self.S_matrix_sparse_tensor,A_tensor)  #Pytorch目前支持的稀疏矩阵乘法只能是Sparse-Dense,其他形式都不行

        # Normlization
        SA_calcu_tensor = MiniTools.normSumAxis1_Tensor(SA_calcu_tensor)
        #SA_calcu_tensor = SA_calcu_tensor / SA_calcu_tensor.sum(axis=1)[:, None]
        # SA_train_tensor[SA_train_tensor != SA_train_tensor] = 0

        # loss_tensor = torch.abs(st_ag_pos_state_tensor - st_pos_state_groundtruth_tensor)
        # mean_loss_tensor = torch.mean(loss_tensor)
        if (loss_type == 'MSE'):
            loss_func = torch.nn.MSELoss()
        elif (loss_type == 'MAE'):
            loss_func = torch.nn.L1Loss()
        # loss = torch.nn.MSELoss()
        # input = torch.randn(3, 5, requires_grad=True)
        # target = torch.randn(3, 5, requires_grad=True)
        # output = loss(input, target)

        # loss = (SA_train_tensor-st_pos_state_groundtruth_tensor)**2
        loss = loss_func(SA_calcu_tensor, SA_GT_tensor)

        # 将tensor转为numpy
        A_array = A_tensor.detach().numpy()
        SA_calcu_df = DataFrame(SA_calcu_tensor.detach().numpy())
        loss_number = loss.detach().numpy()
        self.saveGaussianCurrentResult(A_array.T, SA_calcu_df, loss_number, save_epoch=30)
        loss = loss.requires_grad_()
        return loss

    def optimizePSO(self, swarm_size = 30, max_epoch=5000, loss_type='MAE'):

        # ==============Initilization the Optimization parameters for PSO==================
        # define the limits of the input variables that optimizer is allowed to search within
        # lb:lower-bound ub:upper-bound
        m = self.group_number
        Mean_Dynamic_x1_lb = [0 for x in range(m)]
        Mean_Dynamic_x1_ub = [0.2 for x in range(m)]
        Mean_Dynamic_y1_lb = [0 for x in range(m)]
        Mean_Dynamic_y1_ub = [0.2 for x in range(m)]
        Mean_Dynamic_z1_lb = [0 for x in range(m)]
        Mean_Dynamic_z1_ub = [0.2 for x in range(m)]
        Std_Dynamic_lb = [0 for x in range(m)]
        Std_Dynamic_ub = [1 for x in range(m)]
        # Group_scale_lb = [0 for x in range(m)]
        # Group_scale_ub = [2 for x in range(m)]
        lb = Mean_Dynamic_x1_lb + Mean_Dynamic_y1_lb + Mean_Dynamic_z1_lb + Std_Dynamic_lb  # + Group_scale_lb
        ub = Mean_Dynamic_x1_ub + Mean_Dynamic_y1_ub + Mean_Dynamic_z1_ub + Std_Dynamic_ub  # + Group_scale_ub
        self.lb = lb
        self.ub = ub
        self.optimization_parameter = np.array(lb) + 0.5 * np.array(ub)
        self.uid_state = []
        self.current_iter_num = 0
        self.train_processing = []
        self.eval_processing = []
        self.run_time = 0

        starttime = datetime.datetime.now()
        pso(self.calcuSAbyGaussianParameters, self.lb, self.ub, swarmsize=swarm_size, maxiter=max_epoch)
        endtime = datetime.datetime.now()
        self.run_time = (endtime - starttime).seconds





    def optimizeVAI(self, lr=0.1, max_epoch=100000, loss_type='MAE'):
        # torch.set_default_dtype(torch.double)
        m = self.group_number
        self.train_processing = []
        self.eval_processing = []
        self.run_time = 0

        # 1. form the input
        Mean_Dynamic_x_init = [0.1 for x in range(m)]
        Mean_Dynamic_y_init = [0.1 for x in range(m)]
        Mean_Dynamic_z_init = [0.1 for x in range(m)]
        Std_Dynamic_init = [0.5 for x in range(m)]
        initial_parameter = Mean_Dynamic_x_init + Mean_Dynamic_y_init + Mean_Dynamic_z_init + Std_Dynamic_init
        self.optimization_parameter = initial_parameter

        x = Variable(torch.tensor(initial_parameter), requires_grad=True)

        self.xyz_torch_tensor = torch.tensor(self.xyz_df.values)
        SA_train_tensor = torch.tensor(self.SA_GT_df.values)
        SA_train_tensor = MiniTools.normSumAxis1_Tensor(SA_train_tensor)

        # Convert the scipy.sparse matrix to torch.sparse
        # Note that torch.sparse has no choice but coo !!!
        coo = sparse.coo_matrix(self.S_matrix.T)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        self.S_matrix_sparse_tensor = torch.sparse.ShortTensor(i, v, torch.Size(shape))  # .to_dense()

        optimizer = torch.optim.SGD([x], lr=lr)
        starttime = datetime.datetime.now()
        for epoch in range(max_epoch):
            self.current_iter_num = epoch
            loss = self.calcuSAbyGaussianParametersTensor(x, SA_train_tensor.type(torch.float),loss_type=loss_type)  # 一定是预测值在左，真实值在后
            optimizer.zero_grad()
            # with torch.autograd.detect_anomaly():

            loss.backward()
            optimizer.step()
            print(loss.cpu().detach().numpy())
        self.bananapobGaussianSGDOptimize_parameter = x.detach().numpy()
        self.optimization_parameter = self.bananapobGaussianSGDOptimize_parameter

        endtime = datetime.datetime.now()
        self.run_time = (endtime - starttime).seconds

    def optimizeFNN(self, lr=0.2, max_epoch=1000,loss_type = 'MAE'):
        self.train_processing = []
        self.eval_processing = []
        self.run_time = 0

        # 1. 输入数据初始化
        cuda_flag = 0
        if (cuda_flag == 1):
            xyz_tensor = torch.tensor(self.xyz_df.values, dtype=torch.float32).cuda()
            SA_GT_tensor = torch.tensor(self.SA_GT_df.values.T / self.SA_GT_df.values.T.sum(axis=0),
                                           dtype=torch.float32).T.cuda()
        else:
            xyz_tensor = torch.tensor(self.xyz_df.values, dtype=torch.float32)
            SA_GT_tensor = torch.tensor(self.SA_GT_df.values.T / self.SA_GT_df.values.T.sum(axis=0),
                                           dtype=torch.float32).T
        # Convert the scipy.sparse matrix to torch.sparse
        # Note that torch.sparse has no choice but coo !!!
        coo = sparse.coo_matrix(self.S_matrix.T)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        self.S_matrix_sparse_tensor = torch.sparse.ShortTensor(i, v, torch.Size(shape))  # .to_dense()

        # 2. 模型初始化
        if (cuda_flag == 1):
            net = Activation_Net(in_dim=3, n_hidden_1=16, n_hidden_2=16, out_dim=self.group_number).cuda()
            A_tensor = net(xyz_tensor)
        else:
            net = Activation_Net(in_dim=3, n_hidden_1=16, n_hidden_2=16, out_dim=self.group_number)
            A_tensor = net(xyz_tensor)

        # 3. 训练模型
        # optimizer 是训练的工具
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # hfdd有参数, 学习率

        if (loss_type == 'MSE'):
            loss_func = torch.nn.MSELoss()
        elif (loss_type == 'MAE'):
            loss_func = torch.nn.L1Loss()

        starttime = datetime.datetime.now()
        for epoch in range(max_epoch):
            self.current_iter_num = epoch

            A_tensor = net(xyz_tensor)  # 喂给 net 训练数据 xyz坐标，输出id_ag_pob
            SA_calcu_tensor = torch.sparse.mm(self.S_matrix_sparse_tensor,
                                              A_tensor)  # Pytorch目前支持的稀疏矩阵乘法只能是Sparse-Dense,其他形式都不行
            # Normlization
            SA_calcu_tensor = MiniTools.normSumAxis1_Tensor(SA_calcu_tensor)

            loss = loss_func(SA_calcu_tensor, SA_GT_tensor)

            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

            # 将tensor转为numpy
            A_array = A_tensor.detach().numpy()
            SA_calcu_df = DataFrame(SA_calcu_tensor.detach().numpy())
            loss_number = loss.detach().numpy()
            self.saveGaussianCurrentResult(A_array.T, SA_calcu_df, loss_number, save_epoch=30)

            #print(float(loss.cpu().detach().numpy()))

        # 4. 最优值赋值
        self.age_gender_pob = A_tensor.cpu().detach().numpy()
        endtime = datetime.datetime.now()
        self.run_time = (endtime - starttime).seconds

    def optimizeMultiTaskFNN(self,lr=0.2, max_epoch=3000, loss_type='MAE'):
        self.train_processing = []
        self.eval_processing = []
        self.run_time = 0

        # 1. 输入数据初始化
        cuda_flag = 0
        if (cuda_flag == 1):
            xyz_tensor = torch.tensor(self.xyz_df.values, dtype=torch.float32).cuda()
            SA_GT_tensor = torch.tensor(self.SA_GT_df.values.T / self.SA_GT_df.values.T.sum(axis=0),
                                           dtype=torch.float32).T.cuda()
        else:
            xyz_tensor = torch.tensor(self.xyz_df.values, dtype=torch.float32)
            SA_GT_tensor = torch.tensor(self.SA_GT_df.values.T / self.SA_GT_df.values.T.sum(axis=0),
                                           dtype=torch.float32).T
        # Convert the scipy.sparse matrix to torch.sparse
        # Note that torch.sparse has no choice but coo !!!
        coo = sparse.coo_matrix(self.S_matrix.T)
        values = coo.data
        indices = np.vstack((coo.row, coo.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = coo.shape
        self.S_matrix_sparse_tensor = torch.sparse.ShortTensor(i, v, torch.Size(shape))  # .to_dense()

        # 2. 模型初始化
        if (cuda_flag == 1):
            net = Activation_Multi_Net(in_dim=3, n_hidden_1=6, n_hidden_2=6, out_dim=1).cuda()
            A_tensor = net(xyz_tensor)
        else:
            net = Activation_Multi_Net(in_dim=3, n_hidden_1=12, n_hidden_2=6, out_dim=1)
            A_tensor = net(xyz_tensor)

        # 3. 训练模型
        # optimizer 是训练的工具
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)  # hfdd有参数, 学习率

        if (loss_type == 'MSE'):
            loss_func = torch.nn.MSELoss()
        elif (loss_type == 'MAE'):
            loss_func = torch.nn.L1Loss()

        starttime = datetime.datetime.now()
        for epoch in range(max_epoch):
            self.current_iter_num = epoch

            A_tensor = net(xyz_tensor)  # 喂给 net 训练数据 xyz坐标，输出id_ag_pob
            SA_calcu_tensor = torch.sparse.mm(self.S_matrix_sparse_tensor,
                                              A_tensor)  # Pytorch目前支持的稀疏矩阵乘法只能是Sparse-Dense,其他形式都不行
            # Normlization
            SA_calcu_tensor = MiniTools.normSumAxis1_Tensor(SA_calcu_tensor)

            loss = loss_func(SA_calcu_tensor, SA_GT_tensor)

            optimizer.zero_grad()  # 清空上一步的残余更新参数值
            loss.backward()  # 误差反向传播, 计算参数更新值
            optimizer.step()  # 将参数更新值施加到 net 的 parameters 上

            # 将tensor转为numpy
            A_array = A_tensor.detach().numpy()
            SA_calcu_df = DataFrame(SA_calcu_tensor.detach().numpy())
            loss_number = loss.detach().numpy()
            self.saveGaussianCurrentResult(A_array.T, SA_calcu_df, loss_number, save_epoch=30)

            #print(float(loss.cpu().detach().numpy()))

        # 4. 最优值赋值
        self.age_gender_pob = A_tensor.cpu().detach().numpy()
        endtime = datetime.datetime.now()
        self.run_time = (endtime - starttime).seconds

    def saveTrainProcessing(self, current_object_name):
        DataFrame(self.train_processing).to_csv(self.FL_RESULT_SAVE_PATH + current_object_name + '_train_processing.csv')
        DataFrame(self.eval_processing).to_csv(self.FL_RESULT_SAVE_PATH + current_object_name + '_eval_processing.csv')
        np.savetxt(self.FL_RESULT_SAVE_PATH + current_object_name + '_runtime.txt',
                   np.array([self.run_time]))  # 将数组中数据写入到data.txt文件
        # DataFrame({'run_time':self.run_time}).to_csv(self.save_path + current_object_name + '_runtime.csv')
        np.savetxt(self.FL_RESULT_SAVE_PATH + current_object_name + 'best_parameter.txt',self.optimization_parameter)

    def predictEval(self):
        m=self.group_number
        parameter = self.optimization_parameter
        Mean_Dynamic_x = parameter[0:m]  # 注意，左边闭集，右边开集！
        Mean_Dynamic_y = parameter[m:2 * m]
        Mean_Dynamic_z = parameter[2 * m:3 * m]
        Std_Dynamic = parameter[3 * m:4 * m]

        # Renew the state of every uid
        A = np.zeros((len(self.xyz_df), self.group_number))
        for i in range(m):
            temp_x_square = np.power((self.xyz_df.values[:, 0] - Mean_Dynamic_x[i]), 2)
            temp_y_square = np.power((self.xyz_df.values[:, 1] - Mean_Dynamic_y[i]), 2)
            temp_z_square = np.power((self.xyz_df.values[:, 2] - Mean_Dynamic_z[i]), 2)
            temp_distance_list = np.sqrt((temp_x_square + temp_y_square + temp_z_square))
            temp_A = MiniTools.getGaussianPob(0, Std_Dynamic[i], np.array(
                temp_distance_list))  # for spatial_distance in np.array(temp_distance_list)]
            A[:, i] = temp_A
        return MiniTools.normSumAxis1(A)

    def predictEval(self,x,y,z):
        m=self.group_number
        parameter = self.optimization_parameter
        Mean_Dynamic_x = parameter[0:m]  # 注意，左边闭集，右边开集！
        Mean_Dynamic_y = parameter[m:2 * m]
        Mean_Dynamic_z = parameter[2 * m:3 * m]
        Std_Dynamic = parameter[3 * m:4 * m]

        # Renew the state of every uid
        A = np.zeros(self.group_number)
        for i in range(m):
            temp_x_square = np.power((x - Mean_Dynamic_x[i]), 2)
            temp_y_square = np.power((y - Mean_Dynamic_y[i]), 2)
            temp_z_square = np.power((z - Mean_Dynamic_z[i]), 2)
            temp_distance_list = np.sqrt((temp_x_square + temp_y_square + temp_z_square))
            temp_A = MiniTools.getGaussianPob(0, Std_Dynamic[i], np.array(
                temp_distance_list))  # for spatial_distance in np.array(temp_distance_list)]
            A[i] = temp_A
        return MiniTools.normSumAxis1(A)

if __name__ == "__main__":

    #Test the Pre-processing
    PATH = r'./test-results' #It's only for result save-path, you can assign it win anywhere
    MiniTools.ifFolderExistThenCreate(PATH)
    # PSO_SAVE_PATH = PATH + '/PSO/'
    # MiniTools.ifFolderExistThenCreate(PSO_SAVE_PATH)
    # VI_SAVE_PATH = PATH + '/VI/'
    # MiniTools.ifFolderExistThenCreate(VI_SAVE_PATH)
    # FNN_SAVE_PATH = PATH + '/FNN/'
    # MiniTools.ifFolderExistThenCreate(FNN_SAVE_PATH)
    # MultiTaskFNN_SAVE_PATH = PATH + '/MultiTaskFNN/'
    # MiniTools.ifFolderExistThenCreate(MultiTaskFNN_SAVE_PATH)

    filter_threshold = 30

    #############################This is all the needed inputs#########################
    # 1. This could be derived by 'data_loader.MobakuProcessing.py'
    MOBAKU_SA_GT_PATH = r'..\assets\mobaku_demographic_pkl\by_day_of_week\SA_GT_df.csv'
    # 2. This is the life-pattern feature in 3-D space (from wenjing)
    LIFE_PATTERN_XYZ_PATH = '../assets/xyz_coordinate.csv'
    # 3. This is the S-matrix, could be derived by 'data_loader.PreProcess_Traj.py'
    S_MATRIX_FILE_PATH = '../assets/S-matrix-test/S-matrix.npz'
    # 4. This is the user list, could be derived by 'data_loader.PreProcess_Traj.py'
    INPUT_USER_LIST_PATH = '../assets/S-matrix-20130601_20130701/user_ids_list.npy'
    #################################################################################

    #1. Initializa
    FastLabeling_VI = FastLabeling(MOBAKU_SA_GT_PATH, LIFE_PATTERN_XYZ_PATH, S_MATRIX_FILE_PATH,INPUT_USER_LIST_PATH, FL_RESULT_SAVE_PATH= PATH)
    #FastLabeling_VI.aggregateHour()
    #FastLabeling_VI.filterItem(min_total_number_filter=filter_threshold, age_gender_filter=1)
    #FastLabeling_VI.selectPartID(sampling_rate = 1)
    FastLabeling_VI.filterKeyByNumber(threshold_number=30)

    #2. Train Model
    FastLabeling_VI.optimizeVAI(lr=0.1, max_epoch=3000, loss_type='MAE')
    FastLabeling_VI.saveTrainProcessing('VI')

    #3. Eval model (could load saved model without train!) A is the output
    # This best_parameter file comes from FastLabeling_VI.saveTrainProcessing() function
    FastLabeling_VI.optimization_parameter = np.loadtxt('VIbest_parameter.txt')
    A = FastLabeling_VI.predictEval() #for a large number of users
    x,y,z = 0.1,0.1,0.1
    A = FastLabeling_VI.predictEval(x, y, z) # for a single user
