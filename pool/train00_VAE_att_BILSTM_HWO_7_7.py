import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import torch
import os
import glob
import time
import numpy as np
from fun00_VAE_att_BILSTM_HWO_7_7 import encoder, decoder, vae, GaussianNoise
from random import choice
import torch.nn as nn
import pickle
from sklearn.metrics import mean_squared_error    #, mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_percentage_error
from torch.distributions.categorical  import Categorical

t1 = time.time()

whethertest = '_demo'

save_key = '_baseline00_VAE_BiLSTM_attention'

check_key = '_demo'
batchsize = 100



f_filepath_list = open('filepath_list_7_7'+ whethertest , 'rb')
filepath_list = pickle.load(f_filepath_list)

f_inputs_data = open('inputs_traindata_7_7'+ whethertest , 'rb')
inputs_data = pickle.load(f_inputs_data)

f_outputs_data = open('outputs_traindata_7_7'+ whethertest , 'rb')  ##916232
outputs_data = pickle.load(f_outputs_data)

print(inputs_data.shape, outputs_data.shape)

print(1)

# ##################################################################################################################################
# ##################################################################################################################################

# print('train ready')
# def train(input_dim, linear_dim, output_dim, hidden_dim, latent_dim, batch_size, repeat_times, log_file, val_file, input_days):
#
#
#     vae_ = vae(input_dim, linear_dim, output_dim, hidden_dim, latent_dim, batch_size, repeat_times,input_days).cuda()
#     vae_.train()
#     params = [p for p in vae_.parameters() if p.requires_grad]
#     optimizer = torch.optim.RMSprop(params, lr=1e-3, momentum=0.9, weight_decay=1e-6)  ##优化器 SGD《Momentum《AdaGrad《RMSProp<=Adam
#
#
#     inputs = inputs_data
#     inputs = torch.from_numpy(inputs).cuda()
#     epochs = 100
#     length = inputs.size()[0]   ##total
#     iteration = length/batch_size
#     iteration = int(iteration)
#
#     #--------------------------------
#     outputs = outputs_data
#     outputs = torch.from_numpy(outputs).cuda()
#     #--------------------------------
#
#     ### random shuffle ###   作用：打乱数据顺序，增加随机性，防止局部最优和过拟合
#     random_index = torch.randperm(outputs.size()[0])
#     inputs = inputs[random_index]
#     outputs_tensor = outputs[random_index]
#
#     inputs = inputs.cuda()
#     outputs_tensor = outputs_tensor.cuda()
#     #print(outputs_tensor.shape)
#     lr_ = 1e-3
#     alpha = 0.0000001
#
#     cross_loss = torch.nn.CrossEntropyLoss()
#
#     log = open('./logs/' + log_file, 'w')
#     val_loss = 1e32
#     for epoch in range(epochs):
#         if epoch % 80 == 0:
#             lr_ = lr_ / 10
#             print('learning rate: {}...'.format(lr_))
#             for param_group in optimizer.param_groups:
#                 param_group['lr'] = lr_
#         vae_.train()
#         epoch_loss = 0.
#         epoch_loss_l1 = 0.
#         epoch_loss_kl = 0.
#         #--------------------
#         epoch_val_loss = 0.
#         epoch_val_loss_l1 = 0.
#         epoch_val_loss_kl = 0.
#
#         for i in range(iteration):
#             tmp = int(iteration*0.8)
#             print(epoch,i)
#             if i <= tmp:
#                 inputs_ = inputs[i * batch_size:(i + 1) * batch_size]
#                 gt = outputs_tensor[i*batch_size:(i+1)*batch_size]
#                 mu_, sigma_, pred, _ = vae_(inputs_.float())
#
#
#                 t2 = torch.argmax(gt.permute(0,1,3,2).reshape(-1, 3),dim=1)
#
#                 loss_recon = cross_loss(pred.permute(0,1,3,2).reshape(-1, 3), t2.type(torch.cuda.LongTensor))
#
#                 #loss_recon = torch.mean(torch.sum(- gt * torch.log(pred + 1e-24), 1))
#                 loss_kl = - torch.sum((1 + sigma_ - torch.pow(mu_, 2) - torch.exp(sigma_)))
#                 loss = loss_recon + alpha * loss_kl
#                 epoch_loss += loss.item()/(tmp+1)
#                 epoch_loss_l1 += loss_recon.item()/(tmp+1)
#                 epoch_loss_kl += loss_kl.item()/(tmp+1)
#
#                 optimizer.zero_grad() ## 梯度归零
#                 loss.backward()    ## 反向传播计算得到每个参数的梯度
#                 torch.nn.tools.clip_grad_norm_(vae_.parameters(), 5)
#                 optimizer.step()  ## 通过梯度下降执行一步参数更新
#
#             tmp2 = iteration - int(iteration*0.8)
#             if i > tmp:
#                 vae_.eval()
#                 with torch.no_grad():
#                     inputs_ = inputs[i * batch_size:(i + 1) * batch_size]
#                     gt = outputs_tensor[i*batch_size:(i+1)*batch_size]
#                     mu_, sigma_, pred, _ = vae_(inputs_.float())
#
#                     t2 = torch.argmax(gt.permute(0,1,3,2).reshape(-1, 3),dim=1)
#
#                     loss_recon = cross_loss(pred.permute(0,1,3,2).reshape(-1, 3), t2.type(torch.cuda.LongTensor))
#
#                     loss_kl = - torch.sum((1 + sigma_ - torch.pow(mu_, 2) - torch.exp(sigma_)))
#                     epoch_val_loss_l1 += loss_recon.item()/(tmp2-1)
#                     epoch_val_loss_kl += loss_kl.item()/(tmp2-1)
#
#
#         if epoch_val_loss <= val_loss:
#
#             torch.save(vae_.state_dict(), './logs/' + val_file)
#             print('saving model...',end='')
#             print(epoch_val_loss)
#             val_loss = epoch_val_loss
#             count = 0
#         else:
#             count += 1
#             if count >= 90:  ## 300
#                 print('early stop...')
#                 break
#
#         ### loss == KL
#         log.write('epoch, {}, loss_l1, {}, loss_kl, {}, loss_total, {}\n'.format(epoch, epoch_loss_l1, epoch_loss_kl, epoch_loss))
#         print('epoch, {}, loss_l1, {}, loss_kl, {}, loss_total, {}'.format(epoch, epoch_loss_l1, epoch_loss_kl, epoch_loss))
#
#         print("",end='')
#     log.close()
#     torch.save(vae_.state_dict(), './logs/' + val_file)
#
#
#
# train(input_dim=24, linear_dim=1024, output_dim=24, hidden_dim=128, latent_dim=16, batch_size=100, repeat_times=7, log_file='log_7_7' + whethertest + save_key +'.csv', val_file='val_7_7' + whethertest +save_key+ '.h5',input_days=7)
#
#
#
# print('train finish')

##########################################################################################################################################
##########################################################################################################################################

### load test data

f_inputs_test = open('inputs_testdata_7_7' + whethertest , 'rb')
inputs_test = pickle.load(f_inputs_test)

f_outputs_test = open('outputs_testdata_7_7' + whethertest , 'rb')
outputs_test = pickle.load(f_outputs_test)



print(inputs_test.shape,outputs_test.shape)
print('test ready')
#
# # ############################################################################################################################################
# # ############################################################################################################################################


def test(input_dim, linear_dim, output_dim, hidden_dim, latent_dim, val_file, batch_size, repeat_times,input_days):


    inputs = inputs_test
    outputs  = outputs_test

    inputs_tensor = torch.from_numpy(inputs).cuda()
    outputs_tensor = torch.from_numpy(outputs).cuda()
    length = inputs_tensor.size()[0]

    print(inputs_tensor.shape)

    vae_ = vae(input_dim, linear_dim, output_dim, hidden_dim, latent_dim, batch_size, repeat_times,input_days).cuda()
    vae_.load_state_dict(torch.load('./logs/' + val_file))
    vae_.eval()
    iteration = int(length / batch_size)

    all_traj = []
    for i in range(iteration):
        print(i)
        with torch.no_grad():
            inputs_ = inputs_tensor[i * batch_size:(i + 1) * batch_size]  ## （50,7,3,24）
            mu_, sigma_, pred, lstm1= vae_(inputs_.float(), infer=True)
            noise = GaussianNoise()
            noise1 = noise.sampling(mu_,sigma_)
            pred = vae_.decoder_(lstm1, noise1, True)

        ######## one by one
        prediction_arr2 = pred.cpu().numpy() ## （50,7,3,24）
        outputs_gt = (outputs_tensor[i * batch_size:(i + 1) * batch_size]).cpu().numpy()

        for k in range(0,prediction_arr2.shape[0]):
            for j in range(0,prediction_arr2.shape[1]) :
                print(i,k,j)
                u1_1_day_pre = pd.DataFrame(prediction_arr2[k][j])
                u1_1_day_pre = u1_1_day_pre.T
                u1_1_day_pre.columns = ['H', 'W', "O"]

                ## 一行中最大的
                m = np.zeros_like(u1_1_day_pre.values)
                m[np.arange(len(u1_1_day_pre)), u1_1_day_pre.values.argmax(1)] = 1
                df1 = pd.DataFrame(m)
                u1_1_day_pre['pre_H_max_value'] = df1[0]
                u1_1_day_pre['pre_W_max_value'] = df1[1]
                u1_1_day_pre['pre_O_max_value'] = df1[2]


                ## gt value
                u1_1_day_gt = pd.DataFrame(outputs_gt[k][j])
                u1_1_day_gt = u1_1_day_gt.T
                u1_1_day_gt.columns = ['H', 'W', "O"]

                u1_1_day_pre['gt_H'] = u1_1_day_gt['H']
                u1_1_day_pre['gt_W'] = u1_1_day_gt['W']
                u1_1_day_pre['gt_O'] = u1_1_day_gt['O']

                u1_1_day_pre['iteration'] = i
                u1_1_day_pre['k'] = k
                u1_1_day_pre['j'] = j
                all_traj.append(u1_1_day_pre)


    final_save = pd.concat(all_traj,axis=0)
    final_save.to_csv('test_result_7_7' + whethertest  + save_key + check_key + '.csv', index=False)


# test(input_dim=24, linear_dim=32, output_dim=24, hidden_dim=16, latent_dim=10, batch_size=1, repeat_times=3, val_file='val.h5',test_output_all_user_list=test_output_all_user_list )
test(input_dim=24, linear_dim=1024, output_dim=24, hidden_dim=128, latent_dim=16, batch_size=500, repeat_times=7,val_file='val_7_7' + whethertest  +save_key+ '.h5',input_days=7)


