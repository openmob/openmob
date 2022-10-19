import os.path
import warnings
from japan_holidays import HolidayDataset
import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from joblib import Parallel, delayed
import multiprocessing
from openmob.pool import data_loader

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


class LifePatternProcessor:

    def __init__(self):
        self.kept_data = None
        self.user_id_list = None
        self.raw_gps_file = None
        self.dbscan_min_samples = 10
        self.distance_for_eps = 0.03
        self.map_file = None
        self.clustering_results_folder = None
        self.support_tree_folder = None
        self.raw_gps_folder = None
        self.NMF_results_folder = None
        self.initialize = False
        self.kept_data = None

        if self.initialize:
            self.create_folder()
            self.merge_tree(save_support_tree=True)
            self.NMF_average(save_results=True, raw_gps_file=self.raw_gps_file, raw_gps_folder=self.raw_gps_folder)
            # self.clustering(save_model=True)
            self.plus_home_work_location(save_results=True)
            self.generate_group_HWO_join_area2(save_results=True)

    def create_folder(self):

        if not os.path.exists(self.support_tree_folder):
            os.mkdir(self.support_tree_folder)
        if not os.path.exists(self.NMF_results_folder):
            os.mkdir(self.NMF_results_folder)
        if not os.path.exists(self.clustering_results_folder):
            os.mkdir(self.clustering_results_folder)
        return

    def select_area(self, raw_gps_file=None, map_file=None):

        if raw_gps_file is not None:
            self.raw_gps_file = raw_gps_file

        self.map_file = map_file
        if self.map_file is not None:
            area_map = gpd.GeoDataFrame.from_file(self.map_file)

            if area_map.crs is None:
                area_map.crs = 'epsg:4326'

        ### add if check to load different data source
        data = data_loader.load_tsmc2014_tky_stay_points(self.raw_gps_file)
        self.user_id_list = data.user_id.unique()

        if len(self.user_id_list) == 0:
            print('user_id is null...')
            return
        else:
            stay_data_ = data[['user_id', 'lat', 'lon', 'arrival_time', 'departure_time']]

            stay_data_['arrival_time'] = pd.to_datetime(stay_data_.arrival_time)
            stay_data_['departure_time'] = pd.to_datetime(stay_data_.departure_time)
            stay_data20 = stay_data_.sort_values(by=['user_id', 'arrival_time'])
            stay_data20['lon'] = stay_data20['lon'].astype(float)
            stay_data20['lat'] = stay_data20['lat'].astype(float)

            # delete records which are not located in Japan.
            # stay_data20 = stay_data20[(stay_data20['lon'] >= 125) & (stay_data20['lat'] >= 20)]  # 不在日本境内的被剔除
            stay_data3 = stay_data20.reset_index(drop=True)

            stay_data3['hour'] = pd.to_datetime(stay_data3['arrival_time']).dt.hour
            stay_data3['end_hour'] = pd.to_datetime(stay_data3['departure_time']).dt.hour
            stay_data3['weekday'] = pd.to_datetime(stay_data3['arrival_time']).dt.weekday + 1
            stay_data3['day'] = pd.to_datetime(stay_data3['arrival_time']).dt.day

            stay_data3['time_period'] = pd.to_datetime(stay_data3['departure_time']) - pd.to_datetime(stay_data3['arrival_time'])
            stay_data3['time_period_second'] = (pd.to_datetime(stay_data3['departure_time']) - pd.to_datetime(stay_data3['arrival_time'])).dt.seconds

            if self.map_file is not None:
                geometry = [Point(xy) for xy in zip(stay_data3['lon'], stay_data3['lat'])]
                stay_data4 = gpd.GeoDataFrame(stay_data3, crs="epsg:4326", geometry=geometry)
                stay_data4.crs = 'epsg:4326'
                stay_data5 = gpd.sjoin(stay_data4, area_map, how='left', predicate='intersects', lsuffix='left', rsuffix='right')
                stay_data6 = stay_data5.reset_index(drop=True)
                stay_data6 = stay_data6.drop(['geometry', 'index_right'], axis=1)
                stay_data6['within'] = stay_data6['within'].fillna(-1)
                area_stay = stay_data6[stay_data6['within'] == 1]

                # check the stay ratio of each user, then remove those users who have low stay ratio in the area.
                keep_ids = []
                for ids in self.user_id_list:
                    stay_data6_tmp = stay_data6[stay_data6.user_id == ids]
                    tokyo_stay_tmp = area_stay[area_stay.user_id == ids]
                    if len(stay_data6_tmp) != 0:
                        if (len(stay_data6_tmp) > 10) & (len(tokyo_stay_tmp) >= 4) & (
                                (len(tokyo_stay_tmp) / len(stay_data6_tmp)) >= 0.2):
                            keep_ids.append(ids)
                        else:
                            continue
                    else:
                        continue
                print('total users: {}; kept users: {}'.format(len(self.user_id_list), len(keep_ids)))
                self.kept_data = stay_data6[stay_data6.user_id.isin(keep_ids)]
                self.kept_data = self.kept_data.drop('within', axis=1)
            else:
                self.kept_data = stay_data3
            return self.kept_data

    def detect_home_work(self):

        # if self.kept_data is None:
        #     self.select_area(self.raw_gps_file, self.map_file)
        # assign holidays

        self.kept_data['holiday'] = 0
        self.kept_data.loc[(self.kept_data['weekday'] == 6) | (self.kept_data['weekday'] == 7), 'holiday'] = 1
        japan_holiday = HolidayDataset.HOLIDAYS.keys()
        self.kept_data.loc[self.kept_data.arrival_time.dt.date.isin(japan_holiday), 'holiday'] = 1

        def dbscan_individual(df):

            df = df.reset_index(drop=True)
            row_id = df.index.tolist()
            df['row_id'] = pd.DataFrame(row_id)[0]

            #  DBSCAN cluster for all records ##################################################################

            all_df = df.copy()
            all_df_point = all_df[["lon", "lat"]]
            all_df_for_dbsc = all_df_point.to_numpy(copy=True)  # convert to array   df2.as_matrix().astype("float64", copy=False)
            all_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371), min_samples=self.dbscan_min_samples,
                              algorithm='ball_tree',
                              metric='haversine').fit(np.radians(all_df_for_dbsc))
            all_labels = all_dbsc.labels_
            all_labels_list = all_labels.tolist()
            all_df['all_detect_label'] = pd.DataFrame(all_labels_list)

            # DBSCAN cluster for HOME candidates

            home_df = df.copy()
            home_df1 = home_df[home_df['holiday'] != 1]
            home_df3 = home_df1[
                ((home_df1['hour'] >= 20) & (home_df1['end_hour'] >= 23)) | (home_df1['hour'] <= 4)]  # 粗选home

            home_df4 = home_df3[home_df3['time_period_second'] >= 5400]  # 粗选home
            if home_df4.empty is False:

                home_candidate = home_df4.reset_index(drop=True)
                home_candidate_point = home_candidate[["lon", "lat"]]
                home_candidate_for_dbsc = home_candidate_point.to_numpy(copy=True)  # convert to array   df2.as_matrix().astype("float64", copy=False)
                home_candidate_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371),
                                             min_samples=self.dbscan_min_samples,
                                             algorithm='ball_tree', metric='haversine').fit(np.radians(home_candidate_for_dbsc))
                home_candidate_labels = home_candidate_dbsc.labels_
                home_candidate_labels_list = home_candidate_labels.tolist()
                home_candidate['home_label'] = pd.DataFrame(home_candidate_labels_list)
                home_candidate2 = home_candidate[['row_id', 'home_label']]
                home_df5 = pd.merge(all_df, home_candidate2, left_on='row_id', right_on='row_id', how='left')
            else:
                home_df5 = all_df.copy()
                home_df5['home_label'] = -1

            # DBSCAN cluster for WORK candidates
            work_df = df.copy()

            work_df1 = work_df[work_df['holiday'] != 1]
            work_df2 = work_df1[(work_df1['end_hour'] <= 19) & (work_df1['hour'] >= 8)]
            work_df4 = work_df2[work_df2['time_period_second'] >= 5400]  # 粗选work
            if work_df4.empty is False:

                work_candidate = work_df4.reset_index(drop=True)
                work_candidate_point = work_candidate[["lon", "lat"]]
                work_candidate_for_dbsc = work_candidate_point.to_numpy(copy=True)
                work_candidate_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371), min_samples=self.dbscan_min_samples,
                                             algorithm='ball_tree',
                                             metric='haversine').fit(np.radians(work_candidate_for_dbsc))
                work_candidate_labels = work_candidate_dbsc.labels_

                work_candidate_labels_list = work_candidate_labels.tolist()
                work_candidate['work_label'] = pd.DataFrame(work_candidate_labels_list)
                work_candidate2 = work_candidate[['row_id', 'work_label']]
                final_candidate = pd.merge(home_df5, work_candidate2, left_on='row_id', right_on='row_id', how='left')
            else:
                final_candidate = home_df5.copy()
                final_candidate['work_label'] = -1

            final_candidate.loc[:, ['home_label', 'work_label']] = final_candidate[['home_label', 'work_label']].fillna(-1)

            # determine whether the cluster that obtain from first step is  WORK or HOME
            one_cluster_list = []
            grouped_by_label = final_candidate.groupby(by=['all_detect_label'])
            for key_all, data_all in grouped_by_label:
                one_cluster_label = key_all
                one_cluster_df = data_all.copy()
                if one_cluster_label == -1:
                    one_cluster_df2 = one_cluster_df
                    one_cluster_df2['home_label_new'] = -1
                    one_cluster_df2['work_label_new'] = -1
                    one_cluster_df2['other_label_new'] = -1
                    one_cluster_df2['home_lat'] = -1
                    one_cluster_df2['home_lon'] = -1
                    one_cluster_df2['work_lat'] = -1
                    one_cluster_df2['work_lon'] = -1
                    one_cluster_df2['other_lat'] = -1
                    one_cluster_df2['other_lon'] = -1
                    one_cluster_list.append(one_cluster_df2)
                else:
                    # compare the days
                    home_counts_df = one_cluster_df[one_cluster_df['home_label'] != -1]
                    home_counts_df2 = home_counts_df.drop_duplicates(['day'], keep='first')
                    home_counts1 = pd.value_counts(home_counts_df2['home_label']).sort_values(ascending=False).to_frame()
                    home_counts1['home_index'] = home_counts1.index.tolist()
                    home_counts1.columns = ['num_home', 'home_index']
                    home_counts4 = home_counts1.reset_index(drop=True)
                    work_counts_df = one_cluster_df[one_cluster_df['work_label'] != -1]
                    work_counts_df2 = work_counts_df.drop_duplicates(['day'], keep='first')
                    work_counts1 = pd.value_counts(work_counts_df2['work_label']).sort_values(ascending=False).to_frame()
                    work_counts1['work_index'] = work_counts1.index.tolist()
                    work_counts1.columns = ['num_work', 'work_index']
                    work_counts4 = work_counts1.reset_index(drop=True)

                    if (len(home_counts4) != 0) & (len(work_counts4) != 0):
                        home_max_number = home_counts4.loc[0, 'num_home']
                        home_max_index = home_counts4.loc[0, 'home_index']
                        work_max_number = work_counts4.loc[0, 'num_work']
                        work_max_index = work_counts4.loc[0, 'work_index']
                        if home_max_number >= work_max_number:
                            one_cluster_df2 = one_cluster_df
                            one_cluster_df2['home_label_new'] = home_max_index
                            one_cluster_df2['work_label_new'] = -1
                            one_cluster_df2['other_label_new'] = -1
                            one_cluster_df2['home_lat'] = one_cluster_df2['lat'].mean()
                            one_cluster_df2['home_lon'] = one_cluster_df2['lon'].mean()
                            one_cluster_df2['work_lat'] = -1
                            one_cluster_df2['work_lon'] = -1
                            one_cluster_df2['other_lat'] = -1
                            one_cluster_df2['other_lon'] = -1
                            one_cluster_list.append(one_cluster_df2)
                        elif home_max_number < work_max_number:
                            one_cluster_df2 = one_cluster_df
                            one_cluster_df2['home_label_new'] = -1
                            one_cluster_df2['work_label_new'] = work_max_index
                            one_cluster_df2['other_label_new'] = -1
                            one_cluster_df2['home_lat'] = -1
                            one_cluster_df2['home_lon'] = -1
                            one_cluster_df2['work_lat'] = one_cluster_df2['lat'].mean()
                            one_cluster_df2['work_lon'] = one_cluster_df2['lon'].mean()
                            one_cluster_df2['other_lat'] = -1
                            one_cluster_df2['other_lon'] = -1
                            one_cluster_list.append(one_cluster_df2)

                    elif (len(home_counts4) != 0) & (len(work_counts4) == 0):
                        home_max_index = home_counts4.loc[0, 'home_index']
                        one_cluster_df2 = one_cluster_df
                        one_cluster_df2['home_label_new'] = home_max_index
                        one_cluster_df2['work_label_new'] = -1
                        one_cluster_df2['other_label_new'] = -1
                        one_cluster_df2['home_lat'] = one_cluster_df2['lat'].mean()
                        one_cluster_df2['home_lon'] = one_cluster_df2['lon'].mean()
                        one_cluster_df2['work_lat'] = -1
                        one_cluster_df2['work_lon'] = -1
                        one_cluster_df2['other_lat'] = -1
                        one_cluster_df2['other_lon'] = -1
                        one_cluster_list.append(one_cluster_df2)

                    elif (len(home_counts4) == 0) & (len(work_counts4) != 0):
                        work_max_index = work_counts4.loc[0, 'work_index']
                        one_cluster_df2 = one_cluster_df
                        one_cluster_df2['home_label_new'] = -1
                        one_cluster_df2['work_label_new'] = work_max_index
                        one_cluster_df2['other_label_new'] = -1
                        one_cluster_df2['home_lat'] = -1
                        one_cluster_df2['home_lon'] = -1
                        one_cluster_df2['work_lat'] = one_cluster_df2['lat'].mean()
                        one_cluster_df2['work_lon'] = one_cluster_df2['lon'].mean()
                        one_cluster_df2['other_lat'] = -1
                        one_cluster_df2['other_lon'] = -1
                        one_cluster_list.append(one_cluster_df2)
                    elif (len(home_counts4) == 0) & (len(work_counts4) == 0):
                        one_cluster_df2 = one_cluster_df
                        one_cluster_df2['home_label_new'] = -1
                        one_cluster_df2['work_label_new'] = -1
                        one_cluster_df2['other_label_new'] = one_cluster_label
                        one_cluster_df2['home_lat'] = -1
                        one_cluster_df2['home_lon'] = -1
                        one_cluster_df2['work_lat'] = -1
                        one_cluster_df2['work_lon'] = -1
                        one_cluster_df2['other_lat'] = one_cluster_df2['lat'].mean()
                        one_cluster_df2['other_lon'] = one_cluster_df2['lon'].mean()
                        one_cluster_list.append(one_cluster_df2)

            new_result = pd.concat(one_cluster_list, axis=0)

            # determine home work label order

            new_result_home = new_result[new_result['home_label_new'] != -1]
            if len(new_result_home) != 0:

                new_list = []

                home_new = new_result[new_result['home_label_new'] != -1]
                work_new = new_result[new_result['work_label_new'] != -1]
                other_new = new_result[new_result['other_label_new'] != -1]
                noise_new = new_result[new_result['all_detect_label'] == -1]

                if len(home_new) != 0:
                    home_order_counts = pd.value_counts(home_new['home_label_new']).sort_values(ascending=False).to_frame()
                    home_order_counts['original_home_label'] = home_order_counts.index.tolist()
                    home_order_counts['home_label_new_order'] = list(range(len(home_order_counts)))
                    home_order_counts.columns = ['counts', 'original_home_label', 'home_label_order']

                    home_group = pd.merge(home_new, home_order_counts, left_on='home_label_new',
                                          right_on="original_home_label", how='left')

                    home_group['work_label_order'] = -1
                    home_group['original_work_label'] = -1
                    home_group['other_label_order'] = -1
                    home_group['original_other_label'] = -1

                    home_group_0 = home_group[
                        ['user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon',
                         'hour',
                         'end_hour', 'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id',
                         'all_detect_label', 'home_label', 'work_label', 'home_label_new', 'work_label_new',
                         'other_label_new',
                         'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon', 'work_lat',
                         'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(home_group_0)

                if len(work_new) != 0:
                    work_order_counts = pd.value_counts(work_new['work_label_new']).sort_values(ascending=False).to_frame()
                    work_order_counts['original_work_label'] = work_order_counts.index.tolist()
                    work_order_counts['work_label_new_order'] = list(range(len(work_order_counts)))
                    work_order_counts.columns = ['counts', 'original_work_label', 'work_label_order']

                    work_group = pd.merge(work_new, work_order_counts, left_on='work_label_new',
                                          right_on="original_work_label", how='left')

                    work_group['home_label_order'] = -1
                    work_group['original_home_label'] = -1
                    work_group['other_label_order'] = -1
                    work_group['original_other_label'] = -1

                    work_group_0 = work_group[[
                        'user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon',
                        'hour',
                        'end_hour', 'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id',
                        'all_detect_label', 'home_label', 'work_label', 'home_label_new', 'work_label_new',
                        'other_label_new',
                        'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon', 'work_lat',
                        'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(work_group_0)

                if len(other_new) != 0:
                    other_order_counts = pd.value_counts(other_new['other_label_new']).sort_values(
                        ascending=False).to_frame()
                    other_order_counts['original_other_label'] = other_order_counts.index.tolist()
                    other_order_counts['other_label_new_order'] = list(range(len(other_order_counts)))
                    other_order_counts.columns = ['counts', 'original_other_label', 'other_label_order']

                    other_group = pd.merge(other_new, other_order_counts, left_on='other_label_new',
                                           right_on="original_other_label", how='left')

                    other_group['home_label_order'] = -1
                    other_group['original_home_label'] = -1
                    other_group['work_label_order'] = -1
                    other_group['work_other_label'] = -1

                    other_group_0 = other_group[[
                        'user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon',
                        'hour',
                        'end_hour', 'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id',
                        'all_detect_label', 'home_label', 'work_label', 'home_label_new', 'work_label_new',
                        'other_label_new',
                        'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon', 'work_lat',
                        'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(other_group_0)

                if len(noise_new) != 0:
                    noise_group = noise_new.copy()
                    noise_group['work_label_order'] = -1
                    noise_group['original_work_label'] = -1
                    noise_group['other_label_order'] = -1
                    noise_group['original_other_label'] = -1
                    noise_group['home_label_order'] = -1
                    noise_group['original_home_label'] = -1
                    noise_group_0 = noise_group[[
                        'user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon',
                        'hour',
                        'end_hour', 'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id',
                        'all_detect_label', 'home_label', 'work_label', 'home_label_new', 'work_label_new',
                        'other_label_new',
                        'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon', 'work_lat',
                        'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(noise_group_0)

                final_result = pd.concat(new_list, axis=0)

                final_result_home = final_result[final_result['home_label_order'] != -1]

                if self.map_file is not None:
                    great_tokyo_map = gpd.GeoDataFrame.from_file(self.map_file)
                    great_tokyo_map.crs = 'epsg:4326'

                    geometry = [Point(xy) for xy in zip(final_result_home['home_lon'], final_result_home['home_lat'])]
                    df_final_home = gpd.GeoDataFrame(final_result_home, crs="epsg:4326", geometry=geometry)
                    df_final_home.crs = 'epsg:4326'
                    df_wether_within = gpd.sjoin(df_final_home, great_tokyo_map, how='left', predicate='intersects',
                                             lsuffix='left',
                                             rsuffix='right')
                    df_wether_within['within_right'].fillna(-1)
                    df_final_within = df_wether_within[df_wether_within['within_right'] == 1]
                else:
                    df_final_within = final_result_home

                if len(df_final_within) != 0:
                    final_result2 = final_result.sort_values(by='arrival_time')
                    return final_result2
            return

        final_result = apply_parallel(self.kept_data.groupby('user_id'), dbscan_individual)
        return final_result

    def key_point_detection(self):
        pass

    def merge_tree(self, save_support_tree):
        pass

    def NMF_average(self, save_results, raw_gps_file, raw_gps_folder):
        pass

    def plus_home_work_location(self, save_results):
        pass

    def generate_group_HWO_join_area2(self, save_results):
        pass


def apply_parallel(df_grouped, func):
    ret_lst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(ret_lst)


if __name__ == '__main__':
    lpp_v2 = LifePatternProcessor()
    stay_data2 = lpp_v2.select_area(raw_gps_file='../functions/stay_point_detection/dataset_TSMC2014_TKY_stay_points.csv',
                                   map_file=None)
    container = lpp_v2.detect_home_work()
    print(np.shape(container))
    print('here')
