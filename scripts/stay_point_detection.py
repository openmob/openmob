import glob
import multiprocessing
import time
from math import radians, cos, sin, asin, sqrt

import pandas as pd
import tqdm
from joblib import Parallel, delayed

tqdm.tqdm.pandas()


def data_loader(file_):
    data_ = pd.read_csv(file_, sep=",")
    data_['timestamp'] = pd.to_datetime(data_.M2M_DTTM)
    data_ = data_.sort_values(by=['user_id', 'timestamp'])
    return data_.reset_index(drop=True)[['user_id', 'timestamp', 'lat', 'lon']]


def cal_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c * r * 1000


def cluster_check(tmp_cluster, threshold_dis):
    for i in range(len(tmp_cluster)):
        for j in range(i, len(tmp_cluster)):
            if cal_distance(tmp_cluster[i][0], tmp_cluster[i][1], tmp_cluster[j][0], tmp_cluster[j][1]) > threshold_dis:
                return False
    return True


def stay_point_detection(traj, time_threshold=10, distance_threshold=200):
    traj = traj.reset_index(drop=True)
    num_points = len(traj)
    #     print(num_points)
    time_threshold_ = pd.Timedelta('{}minute'.format(time_threshold))
    sp_ = pd.DataFrame()
    s = pd.DataFrame()
    i = 0
    while i < num_points - 1:
        k = [i]
        j = i + 1

        while j < num_points:
            distance_tmp = cal_distance(lon1=traj.loc[i, 'lon'], lat1=traj.loc[i, 'lat'],
                                        lon2=traj.loc[j, 'lon'], lat2=traj.loc[j, 'lat'])
            if distance_tmp <= distance_threshold:

                if traj.loc[j, 'timestamp'] - traj.loc[i, 'timestamp'] >= time_threshold_:
                    k.append(j)
                    j += 1
                else:
                    i = j
                    break
            else:
                i = j
                break
        else:
            if len(k) >= 2:
                ix = k[0]
                jx = k[-1]
                s.loc[0, 'user_id'] = traj.loc[ix, 'user_id']
                s.loc[0, 'lon'] = traj.loc[ix:jx, 'lon'].mean()
                s.loc[0, 'lat'] = traj.loc[ix:jx, 'lat'].mean()
                s.loc[0, 'arrival_time'] = traj.loc[ix, 'timestamp']
                s.loc[0, 'departure_time'] = traj.loc[jx, 'timestamp']
                sp_ = pd.concat([sp_, s], axis=0)
            break

        if len(k) >= 2:
            ix = k[0]
            jx = k[-1]
            s.loc[0, 'user_id'] = traj.loc[ix, 'user_id']
            s.loc[0, 'lon'] = traj.loc[ix:jx, 'lon'].mean()
            s.loc[0, 'lat'] = traj.loc[ix:jx, 'lat'].mean()
            s.loc[0, 'arrival_time'] = traj.loc[ix, 'timestamp']
            s.loc[0, 'departure_time'] = traj.loc[jx, 'timestamp']
            sp_ = pd.concat([sp_, s], axis=0)
    return sp_


def apply_parallel(df_grouped, func):
    ret_lst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(ret_lst)


if __name__ == '__main__':

    data_list = glob.glob('/data/user/poc2022/rawdata/2022/*.txt.gz')

    for file in data_list:
        print(time.ctime(), file)
        output_name = file.split('/')[-1][:4] + file.split('/')[-1][-11:-7]

        chunks = pd.read_csv(file, sep=",", chunksize=10000000)
        try:
            for data in chunks:
                data['M2M_DTTM'] = pd.to_datetime(data.M2M_DTTM)
                data = data.sort_values(by=['M2M_MODEL_CD', 'M2M_DTTM'])
                data = data.reset_index(drop=True)[['M2M_MODEL_CD', 'M2M_DTTM',
                                                    'LATITUDE', 'LONGITUDE', 'ALL_RUN_DIST']]

                SP = apply_parallel(data.groupby('M2M_MODEL_CD'), stay_point_detection)
                SP.to_csv('../data/' + output_name + '.csv', mode='a')
        except OSError:
            #         print('error found in ', file)
            file1 = open("broken_files.txt", "a")  # append mode
            file1.write(file)
            file1.write('\n')
            file1.close()
