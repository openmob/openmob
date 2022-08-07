import pandas as pd
import csv
import numpy as np
import os

def get_traj(filename):#path,
    # time = 0
    full_data = []
    for time in range(0,24):
        target_len = 20
        traj = pd.read_csv(filename,header=None)#path+
        tm_data = traj[traj[0].isin([time])]
        tm_data = tm_data.reset_index(drop=True)
        tm_data = tm_data.drop([0],axis=1)
        #full_data = []
        for i in range(len(tm_data)):
            slice = tm_data.iloc[i]
            str = slice[1]
            str = str.replace('[','').replace(']','')
            l = str.split(',')
            l = list(map(int, l))
            tmp = []
            for item in l:
                item = item#/400
                tmp.append(item)
            l = tmp
            if len(l) < target_len:
                if len(l)>target_len:
                    continue
                if len(l)<target_len:
                    start = l[0]
                    end = l[-1]
                    sub = (target_len-len(l))//2
                    for ii in range(sub):
                        l.insert(0,start)
                    for jj in range(sub):
                        l.append(end)
                    if len(l)>target_len:
                        l.pop()
                    if len(l)<target_len:
                        l.append(end)
                #print(len(l))
                full_data.append(l)
            else:
                full_data.append(l[:target_len])

    #return np.array(full_data)
    df = pd.DataFrame(data=np.array(full_data))
    #df.to_csv('E:/Didi/20_len_patten/' + filename, header=False, index=False)
    df.to_csv('patten_len_20.csv',header=False,index=False)
    #np.save('traj_data.npy',file)

def re_sample(filename):#path,
    data = pd.read_csv(filename, header=None)#path+
    sample_list = []
    for i in range(len(data)):
        line = data.iloc[i]
        sample_list.append([line[0],line[5],line[8],line[10],line[13],line[19]])

    df = pd.DataFrame(data=sample_list)
    #df.to_csv('E:/Didi/6_len_patten/'+filename,header=False,index=False)
    df.to_csv('6_sample.csv', header=False, index=False)

def onehot(filename):#path,
    num_class = 64#400
    inputs = pd.read_csv(filename, header=None).values#path+
    num, length = inputs.shape
    # 179, 20
    onehot = np.zeros((num, length, num_class))

    for i in range(num):  # 抽取一行
        for j in range(length):  # 这一行从左往右

            onehot[i, j, inputs[i, j]] = 1
        #print()
    filename = filename.split('.')[0]
    #np.save('E:/Didi/onehot/'+filename+'.npy', onehot)
    np.save('onehot.npy', onehot)
    #return onehot

get_traj('full_pattern_8_8.csv')
re_sample('patten_len_20.csv')
onehot('patten_len_20.csv')

#data_path = 'E:/Didi/full_patten/'
# data_path = 'E:/Didi/20_len_patten/'
# for root, dirs, files in os.walk(data_path):
#     for filename in files:
#         print(data_path+filename)
#         #get_traj(data_path,filename)
#         #re_sample(data_path,filename)
#         #onehot(data_path,filename)





