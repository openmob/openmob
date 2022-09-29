import argparse
import pandas as pd
import sys
import time
import os
# from os.path import abspath, dirname
# os.chdir(dirname(abspath(__file__)))
# sys.path.insert(0, '../../utils/')
# print(os.getcwd())
from openmob.utils.stay_point_detection import *


def load_data_tsmc(input_file):
    data = pd.read_table(input_file, encoding='latin-1', header=None)
    data.columns = ['user_id', 'venue_id', 'venue_category_id', 'venue_name', 'lat', 'lon', 'time_zone_offset',
                    'utc_time']
    return data
def check_output_folder():
    if not os.path.exists('./stay_points/'):
        os.mkdir('./stay_points/')
    else:
        print('output folder existed...')

def timestamp_calc(line):
    timestamp = pd.to_datetime(line.utc_time) + pd.Timedelta('{}.minutes'.format(line.time_zone_offset))
    return timestamp


def separate_trip(input_file, length):
    check_output_folder()
    data = load_data_tsmc(input_file)
    total_user_number = len(data.user_id.unique())
    if length >= total_user_number:
        print('Desired number of output separate files is larger than total user number...')
    for user_id_ in data.user_id.unique()[:length]:
        tmp = data[data.user_id == user_id_]
        timestamp = tmp.T.apply(lambda x: timestamp_calc(x))
        tmp = pd.concat([tmp, timestamp.rename('timestamp')], axis=1)
        # tmp.to_csv('./{}.csv'.format(user_id_), index=False)
        tmp = stay_point_detection_process(tmp)
        tmp.to_csv('./stay_points/{}.csv'.format(user_id_), index=False)
    return


def stay_point_detection_process(data):
    print(time.ctime())
    try:
        data = data.sort_values(by=['user_id', 'timestamp'])
        data = data.reset_index(drop=True)[['user_id', 'timestamp',
                                            'lat', 'lon', 'venue_name']]
        sp = apply_parallel(data.groupby('user_id'), stay_point_detection)
        return sp
    except OSError:
        print('error found...')
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-f',
                        default='../../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--length', '-l', default=10, type=int,
                        help='desired number of output separate files')
    args = parser.parse_args()

    separate_trip(input_file=args.input_file, length=args.length)
