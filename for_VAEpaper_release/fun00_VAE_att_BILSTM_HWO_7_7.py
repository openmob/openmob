import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import glob
import os
import time


class encoder(nn.Module):

    def __init__(self, hidden_dim, input_dim, output_dim, latent_dim, batch_size, input_days):
        super(encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_days = input_days

        self.kernel_size = 3
        self.lstm = nn.LSTM(self.output_dim, self.hidden_dim, bias=True, batch_first=True,
                            bidirectional=True)  ## batch_first=True, 输入维度为(batch_size,序列长度seq_len,输入维数input_size)
        self.linear_in = nn.Linear(self.input_dim, self.output_dim, bias=True)
        self.linear_out = nn.Linear(self.output_dim, self.output_dim, bias=True)

        # self.conv1 = nn.Conv2d(in_channels = self.input_days , out_channels = self.input_days , kernel_size = (self.kernel_size,self.kernel_size), stride=1, padding=0, dilation=1, groups=1, bias=True) ## 或者5*5 * Kernel size can't be greater than actual input size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3, 3), stride=1, padding=0, dilation=1,
                               groups=1, bias=True)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=1, padding=0, dilation=1,
                               groups=1, bias=True)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=(1, 3), stride=1, padding=0, dilation=1,
                               groups=1, bias=True)

        self.hidden_sim_re = 2 * self.hidden_dim * self.input_days

        self.mu = nn.Linear(self.hidden_sim_re, self.latent_dim, bias=True)
        self.log_sigma = nn.Linear(self.hidden_sim_re, self.latent_dim, bias=True)

        self.relu = nn.ReLU()
        self.LeakyreLU = nn.LeakyReLU()
        self.softsign = nn.Softsign()

        self.hidden_init = (
        torch.autograd.Variable(torch.zeros(2 * 1, self.batch_size, self.hidden_dim).to(torch.device('cuda'))),
        torch.autograd.Variable(torch.zeros(2 * 1, self.batch_size, self.hidden_dim)).to(torch.device('cuda')))

    def forward(self, trajectory):  ### 原来的输入trajectory(200,7,3,24)

        in_list_7_days = []
        for days in range(0, self.input_days):
            aa = trajectory[:, days, :, :]  ## 得到(200,3,24)
            bb = aa.unsqueeze(1)  ## 得到(200,1,3,24)

            locrep = self.conv1(bb).cuda()  ## 得到(200,16,1,22)
            locrep = self.LeakyreLU(locrep).cuda()
            locrep = self.conv2(locrep).cuda()  ## 得到（200,32,1,18）
            locrep = self.LeakyreLU(locrep).cuda()
            locrep = self.conv3(locrep).cuda()  ## 得到(200,64,1,16)
            locrep = self.LeakyreLU(locrep).cuda()
            locrep_flatten = locrep.flatten(1)  ##得到(1000,4*16*20),即(1000,1120) 得到（200，1024）

            in_list_7_days.append(locrep_flatten)

        cov_concat = torch.stack(in_list_7_days, dim=1).cuda()  ### 得到(200,7,1120)   得到200,7,1024

        locrep_down = self.LeakyreLU(cov_concat).cuda()  ##得到(200,7,1024) ## 激活函数（一般紧跟conv后面,让它不要线性变化）：Softsign / SoftPlus/ sogmoid / tanh /ReLU / Leaky-ReLU / ELU / Maxout

        # down = locrep[:,:,-1] ## 改后两维 reshape

        lstm1, (hn, cn) = self.lstm(locrep_down,self.hidden_init)  ##得到(1000,7,128) LSTM输入的三个维度(batch_size,序列长度seq_len,输入维数input_size)   #### 问题： LSTM为啥放在这？
        ## Bi(200,7,256)
        # 为了进行采样再次进行降维3维->2维，
        lstm0 = lstm1.reshape(lstm1.size()[0], lstm1.size()[1] * lstm1.size()[2])  ## 得到（200,896） BI(200,1792)

        #### 问题：为啥没有池化层？
        mu_ = self.mu(lstm0)  ## 对mu 全连层  ##(200,32)
        mu_ = self.softsign(mu_)  ## 对mu 激活函数层
        sigma_ = self.log_sigma(lstm0)  ##对mu 全连接层  ##(200,32)
        sigma_ = self.softsign(sigma_)  ## 对sigma 激活层


        return mu_, sigma_, lstm1, cn

class att(nn.Module):
    def __init__(self):
        super(att, self).__init__()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, lstm_enc, lstm_dec):
        lstm_dec_att = []
        for i in range(lstm_dec.size()[1]):

            if len(lstm_dec_att) == 0:
                lstm_one_dec = lstm_dec[:, i:i+1, :]
                lstm_mul_dec = lstm_one_dec.expand(lstm_dec.size()[0], lstm_dec.size()[1], lstm_dec.size()[2])
                score_lstm = torch.mul(lstm_mul_dec, lstm_enc)
                score_lstm = torch.sum(score_lstm, 2)
                att_lstm = self.softmax(score_lstm)
                att_lstm = torch.unsqueeze(att_lstm, 2)
                att_lstm = att_lstm.expand(lstm_dec.size()[0], lstm_dec.size()[1], lstm_dec.size()[2])
                lstm_one_att = torch.sum(torch.mul(lstm_enc, att_lstm), 1) + lstm_dec[:, i, :]
                lstm_dec_att.append(torch.unsqueeze(lstm_one_att, 1))
            else:
                lstm_one_dec = lstm_dec[:, i:i+1, :]
                lstm_mul_dec = lstm_one_dec.expand(lstm_dec.size()[0], lstm_dec.size()[1]+i, lstm_dec.size()[2])
                t = torch.cat(lstm_dec_att, 1)
                lstm_enc_cat = torch.cat((lstm_enc, torch.cat(lstm_dec_att, 1) ), 1)
                score_lstm = torch.mul(lstm_mul_dec, lstm_enc_cat)
                score_lstm = torch.sum(score_lstm, 2)
                att_lstm = self.softmax(score_lstm)
                att_lstm = torch.unsqueeze(att_lstm, 2)
                att_lstm = att_lstm.expand(lstm_dec.size()[0], lstm_dec.size()[1]+i, lstm_dec.size()[2])
                lstm_one_att = torch.sum(torch.mul(lstm_enc_cat, att_lstm), 1) + lstm_dec[:, i, :]
                lstm_dec_att.append(torch.unsqueeze(lstm_one_att, 1))

        # t = torch.cat(lstm_dec_att, 1)
        return torch.cat(lstm_dec_att, 1)



class decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, latent_dim, batch_size, repeat_times, input_days):
        super(decoder, self).__init__()
        self.rep = repeat_times
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.input_days = input_days

        # self.hidden_dim = 22

        # self.lstm = nn.LSTM(self.latent_dim, self.hidden_dim, batch_first=True, bias=True)
        # self.lstm = nn.LSTM(int(self.latent_dim /7), self.hidden_dim, batch_first=True, bias=True)
        self.lstm = nn.LSTM(self.latent_dim, self.hidden_dim, batch_first=True, bias=True, bidirectional=True)
        # self.lstm = nn.LSTMCell(self.hidden_dim, self.hidden_dim, bias=True)

        self.linear = nn.Linear(self.hidden_dim, self.output_dim, bias=True)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(1, 3), stride=(1, 1),
                                          padding=(0, 0), dilation=1,
                                          groups=1, bias=True)

        self.deconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(1, 5), stride=(1, 1),
                                          padding=(0, 0), dilation=1,
                                          groups=1, bias=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                                          padding=(0, 0), dilation=1,
                                          groups=1, bias=True)

        # self.deconv2 = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(3,3), stride=(1,1), padding=(0,0), dilation=1,
        #                        groups=1, bias=True)
        self.relu = nn.ReLU()
        self.LeakyreLU = nn.LeakyReLU()
        self.softsign = nn.Softsign()
        self.softmax = nn.Softmax(dim=2)  ##这里sofemax传递的方向容易错

        self.hidden_init = (
        torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim).to(torch.device('cuda'))),
        torch.autograd.Variable(torch.zeros(2, self.batch_size, self.hidden_dim)).to(torch.device('cuda')))

        self.embedrep = nn.Linear(self.latent_dim, self.latent_dim * self.input_days)  # self.hidden_dim*1)#

        self.embedding_lstm = nn.Linear(2 * self.hidden_dim, 1024)
        self.att_net = att().cuda()

    def forward(self, lstm_enc, gaussian_noise, infer):  ### 原始的gaussian_noise输入(200,32)

        ### 7-7
        gaussian_noise1 = self.embedrep(gaussian_noise)  ## 得到（200.224）
        gaussian_noise1 = self.softsign(gaussian_noise1).cuda()
        repvec = gaussian_noise1.reshape(self.batch_size, self.input_days,
                                         int(gaussian_noise1.size()[1] / 7))  ##得到(200,7,32)

        lstm1, _ = self.lstm(repvec,
                             self.hidden_init)  # (hn,cn))### 得到(200,7,128) BI(200,7,256  LSTM输入的三个维度(batch_size,序列长度seq_len,输入维数input_size)

        lstm_att_dec =  self.att_net(lstm_enc, lstm1)


        lstm_att_dec = self.embedding_lstm(lstm_att_dec)  ## 得到（200,7，1024）
        lstm1_up = lstm_att_dec.reshape(lstm_att_dec.size()[0], lstm_att_dec.size()[1], 64, 1, 16)  ## 得到（200,7,64,1,16）
        lstm1_up = self.softsign(lstm1_up).cuda()

        out_list_7_days = []
        for days in range(0, self.input_days):
            dd = lstm1_up[:, days, :, :, :]  ## 得到(200,64，1,16)
            # dd = cc.unsqueeze(1)   ## 得到(200,1,1,22)
            locrep = self.deconv3(dd).cuda()  ##（200,32,1,18）
            locrep = self.LeakyreLU(locrep).cuda()
            locrep = self.deconv1(locrep).cuda()  ## 得到（200，16,1,22）
            locrep = self.LeakyreLU(locrep).cuda()
            locrep = self.deconv2(locrep).cuda()  ##  得到（200,1,3,24）
            locrep = self.LeakyreLU(locrep).cuda()
            locrep_out = self.softmax(locrep).cuda() ## 激活层，得到(1000,1,3,24)

            out_list_7_days.append(locrep_out)

        out_cov_concat = torch.stack(out_list_7_days, dim=1).cuda()  ### 得到(200,7,1,3,24)
        output = out_cov_concat[:, :, -1, :, :]  ## 得到(200,7,3,24)
        # output = self.softsign(out_cov_concat).cuda() ## 得到(200,7,3,24)

        return output


class GaussianNoise():

    def __init__(self):
        self.mean_ = 0.
        self.std_ = 1.

    def sampling(self, mu_, sigma_):  # caiyang
        epsilon = torch.empty(mu_.shape[0], mu_.shape[1]).normal_(mean=self.mean_, std=self.std_).cuda()
        return mu_ + torch.exp(sigma_ / 2) * epsilon
        # return epsilon

    def weighted_sampling(self, mu_, sigma_, weight_):
        index_ = list(torch.utils.data.WeightedRandomSampler(weight_.view(-1), int(weight_.sum()), replacement=True))
        mu_all = mu_[index_]
        sigma_all = sigma_[index_]
        epsilon_all = torch.empty(mu_all.shape[0], mu_all.shape[1]).normal_(mean=self.mean_, std=self.std_)
        return index_, mu_all + torch.exp(sigma_all / 2) * epsilon_all


class vae(nn.Module):
    def __init__(self, input_dim, linear_dim, output_dim, hidden_dim, latent_dim, batch_size, repeat_times, input_days):
        super(vae, self).__init__()

        self.input_dim = input_dim
        self.linear_dim = linear_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.rep = repeat_times
        self.input_days = input_days

        self.encoder_ = encoder(self.hidden_dim, self.input_dim, self.linear_dim, self.latent_dim, self.batch_size,
                                self.input_days).cuda()
        self.decoder_ = decoder(self.hidden_dim, self.output_dim, self.latent_dim, self.batch_size, self.rep,
                                self.input_days).cuda()

        self.gaussian_noise = GaussianNoise()

    def forward(self, inputs, infer=False):
        mu_, sigma_, lstm1, cn = self.encoder_(inputs)  ## (1000,10)
        random_noise = self.gaussian_noise.sampling(mu_.cuda(), sigma_.cuda())  ### (1000,10)
        outputs = self.decoder_(lstm1, random_noise, infer)  ###(1000,3,24)

        return mu_, sigma_, outputs, lstm1
