from openmob.utils.life_pattern_processor import LifePatternProcessor
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Args for extracting life pattern from GPS trajectory datasets.')
    parser.add_argument('--input_file', '-if',
                        default='../functions/stay_point_detection/dataset_TSMC2014_TKY_stay_points.csv',
                        help='file of GPS trajectory stay points.')
    parser.add_argument('--clustering_results_folder', '-crf', default='./clustering_results/')
    parser.add_argument('--support_tree_folder', '-stf', default='./support_tree/')
    parser.add_argument('--nmf_results_folder', '-nrf', default='./nmf_results/')
    parser.add_argument('--dbscan_min_samples', '-dms', default=1)
    parser.add_argument('--distance_for_eps', '-dfe', default=0.03)

    args = parser.parse_args()

    lpp = LifePatternProcessor(
        clustering_results_folder=args.clustering_results_folder,
        support_tree_folder=args.support_tree_folder,
        nmf_results_folder=args.nmf_results_folder,
        dbscan_min_samples=args.dbscan_min_samples,
        distance_for_eps=args.distance_for_eps
                               )

    lpp.select_area(raw_gps_file=args.input_file, map_file=None)
    lpp.detect_home_work()
    lpp.extract_life_pattern()
    lpp.support_tree()
    lpp.merge_tree(save_support_tree=True)
    lpp.pattern_probability_matrix()
    lpp.nmf_average()
