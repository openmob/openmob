import os
import pandas as pd
import datetime
import math
# 将轨迹真值转化为patten并保存到文件
# 保留所有途径网格
def calc_lable(lat,long):
    size = 20
    x1 = 108.92
    x2 = 109.01
    y1 = 34.28
    y2 = 34.20
    xx = (x2 - x1) / size
    yy = (y1 - y2) / size
    num_x = math.ceil((float(long) - x1) / xx)  # 第num_x列
    num_y = size - math.floor((float(lat) - y2) / yy)  # 第num_y行
    label = (num_y - 1) * size + num_x
    return label

def traj2patten(data_path,filename):
    data = pd.read_csv(data_path+filename,header=None)
    # data = data.sort_values(by=[0,1,2])
    all_traj = []
    curr_traj = []

    prev_driver_id = data[0][0]
    prev_order_id = data[1][0]
    prev_time = datetime.datetime.utcfromtimestamp(data[2][0])
    prev_long = float(data[3][0])
    prev_lat = float(data[4][0])
    #prev_lable = calc_lable(prev_lat,prev_long)
    for i in range(len(data)):
        curr_driver_id = data[0][i]
        curr_order_id = data[1][i]
        curr_time = datetime.datetime.utcfromtimestamp(data[2][i])
        curr_long = float(data[3][i])
        curr_lat = float(data[4][i])
        #print(driver_id,order_id,time,long,lat)
        # 当前轨迹信息为空，加入轨迹的第一个点
        if not curr_traj:
            prev_driver_id = curr_driver_id
            prev_order_id = curr_order_id
            prev_time = curr_time
            prev_long = curr_long
            prev_lat = curr_lat
            curr_traj.append([curr_long,curr_lat,curr_time.hour])
        # 当前轨迹不为空
        else:
            # 是当前轨迹的下一个坐标点信息
            if prev_driver_id == curr_driver_id and prev_order_id == curr_order_id:
                curr_traj.append([curr_long, curr_lat, curr_time.hour])
            # 开始下一条新的轨迹
            else:
                # curr_traj.append([prev_long, prev_lat, prev_time.hour])
                # 加入终点的信息
                all_traj.append(curr_traj)
                curr_traj = []
                curr_traj.append([curr_long, curr_lat, curr_time.hour])

        prev_driver_id = curr_driver_id
        prev_order_id = curr_order_id
        prev_time = curr_time
        prev_long = curr_long
        prev_lat = curr_lat
    # 最后的点
    curr_traj.append([curr_long, curr_lat, curr_time.hour])
    all_traj.append(curr_traj)
    all_pattern = []

    print(len(all_traj))
    for item in all_traj:
        lab_list = []
        t = item[0][2]
        prev_lab = calc_lable(item[0][1], item[0][0])
        for i in item:
            curr_lab = calc_lable(i[1], i[0])
            if curr_lab < 0 or curr_lab > 400:
                continue
            if prev_lab != curr_lab:
                lab_list.append(curr_lab)
                prev_lab = curr_lab
        if len(lab_list) > 1:
            all_pattern.append([t, lab_list])
    # for pat in all_pattern:
    #     print(pat)
    df = pd.DataFrame(data=all_pattern)
    #df = df.drop_duplicates(keep='first')
    df = df.sort_values(by=[0])
    print(len(df))
    df.to_csv('E:/LocationMind/full_patten/'+filename,header=False,index=False)


data_path = 'E:/LocationMind/sorted_data/'
for root, dirs, files in os.walk(data_path):
    for filename in files:
        print(data_path+filename)
        traj2patten(data_path,filename)