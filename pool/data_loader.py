import argparse
import os.path

import pandas as pd


def timestamp_calc(line):
    timestamp = pd.to_datetime(line.utc_time) + pd.Timedelta('{}.minutes'.format(line.time_zone_offset))
    return timestamp


def load_tsmc2014_tky(input_file):
    if not os.path.exists(input_file):
        df = pd.read_table('../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt', encoding='latin-1', header=None)
        df.columns = ['user_id', 'venue_id', 'venue_category_id', 'venue_name', 'lat', 'lon', 'time_zone_offset',
                  'utc_time']
        timestamp = df.T.apply(lambda x: timestamp_calc(x))
        df = pd.concat([df, timestamp.rename('timestamp')], axis=1)
        df.to_csv(input_file, index=False)
        return df
    else:
        df = pd.read_csv(input_file)
        return df


def load_tsmc2014_tky_stay_points(input_file):
    if not os.path.exists(input_file):
        print('No exited stay points, try stay_point_detection_process at first..')
        return
    else:
        df = pd.read_csv(input_file)
        return df


def individual_traj(input_file, length, output_folder):
    data = load_tsmc2014_tky(input_file)
    total_uid = data.user_id.unique()

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    if length > len(total_uid):
        print('length defined is larger than total user ids..')
    for i in range(len(total_uid[:length])):
        print(i)
        tmp = data[data.user_id == total_uid[i]]
        tmp.to_csv(output_folder + '{}.csv'.format(total_uid[i]), index=False)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.csv',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--length', '-l',
                        default=10, type=int,
                        help='length of extracted users.')
    parser.add_argument('--output_folder', '-of',
                        default='../functions/life_pattern_extraction/individual_traj/',
                        help='output folder of extracted individual trajectory.')
    args = parser.parse_args()

    # data = load_tsmc2014(args.input_file)
    individual_traj(args.input_file, args.length, args.output_folder)
