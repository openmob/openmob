import argparse
import os.path

import pandas as pd

"""
We will consider multiple GPS data sources in the future.
"""


def timestamp_calc(line):
    timestamp = pd.to_datetime(line.utc_time) + pd.Timedelta('{}.minutes'.format(line.time_zone_offset))
    return timestamp


def load_tsmc2014_tky(input_file):
    if not os.path.exists(input_file):
        df = pd.read_table(input_file.replace('csv', 'txt'), encoding='latin-1', header=None)
        df.columns = ['user_id', 'venue_id', 'venue_category_id', 'venue_name', 'lat', 'lon', 'time_zone_offset',
                      'utc_time']
        timestamp = df.T.apply(lambda x: timestamp_calc(x))
        df = pd.concat([df, timestamp.rename('timestamp')], axis=1)
        df.to_csv(input_file, index=False)
        df['timestamp'] = pd.to_datetime(df.timestamp)
        return df
    else:
        df = pd.read_csv(input_file)
        df['timestamp'] = pd.to_datetime(df.timestamp)
        return df


def load_tsmc2014_tky_stay_points(input_file):
    if not os.path.exists(input_file):
        print('No exited stay points, try stay_point_detection_process at first..')
        return
    else:
        df = pd.read_csv(input_file)
        return df


def check_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    else:
        print('output folder existed...')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.csv',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--length', '-l',
                        default=10, type=int,
                        help='length of extracted users.')
    parser.add_argument('--output_folder', '-of',
                        default='../functions/life_pattern_processing/individual_traj/',
                        help='output folder of extracted individual trajectory.')
    args = parser.parse_args()

    data = load_tsmc2014_tky(args.input_file)
    print('here')
