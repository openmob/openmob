from functions import stay_point_detection_process
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for separating trips from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='./datasets/dataset_tsmc2014/dataset_TSMC2014_TKY.txt',
                        help='file of GPS trajectory datasets.')
    parser.add_argument('--length', '-l', default=10, type=int,
                        help='desired number of output separate files')
    parser.add_argument('--output_folder', '-of',
                        default='./functions/stay_point_detection/stay_points/',
                        help='file of GPS trajectory datasets.')
    args = parser.parse_args()

    stay_point_detection_process.separate_trip(input_file=args.input_file, length=args.length, output_folder=args.output_folder)