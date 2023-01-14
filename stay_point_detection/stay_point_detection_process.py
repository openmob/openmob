import os
import sys
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools import stay_point_detection
from visualization import visualization_stay_points

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.csv',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--time_threshold', '-th',
                        default=10,
                        help='time threshold for StayPointDetector.')
    parser.add_argument('--distance_threshold', '-dh',
                        default=200,
                        help='distance threshold for StayPointDetector.')
    parser.add_argument('--worker', '-w',
                        default=4,
                        help='number of cores used.')
    parser.add_argument('--output_folder', '-of',
                        default='./',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--visualize', default=True,
                        help='a simple visualization of detected stay points.')
    args = parser.parse_args()

    sp, data = stay_point_detection.stay_point_detection_process(args)
    if args.visualize:
        m = visualization_stay_points(data, sp, number=3)
        m.save('./stay_points.html')
