import argparse
import pandas as pd
import sys
import time
import os
from openmob.pool.stay_point_detection import *





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='../../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--length', '-l', default=10, type=int,
                        help='desired number of output separate files')
    parser.add_argument('--output_folder', '-of',
                        default='./stay_points/',
                        help='file of GPS trajectory datasets.')
    args = parser.parse_args()

    separate_trip(input_file=args.input_file, length=args.length, output_folder=args.output_folder)
