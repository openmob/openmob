import argparse
from openmob.utils import stay_point_detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='../../datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.csv',
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
    args = parser.parse_args()

    stay_point_detection.stay_point_detection_process(args)
