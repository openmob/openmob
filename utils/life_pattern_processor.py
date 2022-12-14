import multiprocessing
import os.path
import warnings

import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
from sklearn.decomposition import NMF

import japan_holidays
from openmob.pool import data_loader

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


class LifePatternProcessor:

    def __init__(self,
                 clustering_results_folder='./clustering_results/',
                 support_tree_folder='./support_tree/',
                 nmf_results_folder='./nmf_results/',
                 dbscan_min_samples=1,
                 distance_for_eps=0.03,
                 ):
        self.life_pattern = None
        self.pattern_probability_mat = None
        self.merged_tree = None
        self.tree_concat = None
        self.home_work_result = None
        self.kept_data = None
        self.user_id_list = None
        self.raw_gps_file = None
        self.dbscan_min_samples = dbscan_min_samples
        self.distance_for_eps = distance_for_eps
        self.map_file = None
        self.clustering_results_folder = clustering_results_folder
        self.support_tree_folder = support_tree_folder
        self.raw_gps_folder = None
        self.NMF_results_folder = nmf_results_folder
        self.initialize = False
        self.kept_data = None

        # if self.initialize:
        self.create_folder()
        #     self.merge_tree(save_support_tree=True)
        #     self.NMF_average(save_results=True, raw_gps_file=self.raw_gps_file, raw_gps_folder=self.raw_gps_folder)
        #     # self.clustering(save_model=True)
        #     self.plus_home_work_location(save_results=True)
        #     self.generate_group_HWO_join_area2(save_results=True)

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

        # add if check to load different data source
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

            stay_data3 = stay_data20.reset_index(drop=True)

            stay_data3['hour'] = pd.to_datetime(stay_data3['arrival_time']).dt.hour
            stay_data3['end_hour'] = pd.to_datetime(stay_data3['departure_time']).dt.hour
            stay_data3['weekday'] = pd.to_datetime(stay_data3['arrival_time']).dt.weekday + 1
            stay_data3['day'] = pd.to_datetime(stay_data3['arrival_time']).dt.day

            stay_data3['time_period'] = pd.to_datetime(stay_data3['departure_time']) - pd.to_datetime(
                stay_data3['arrival_time'])
            stay_data3['time_period_second'] = (pd.to_datetime(stay_data3['departure_time']) - pd.to_datetime(
                stay_data3['arrival_time'])).dt.seconds

            if self.map_file is not None:
                area_map = gpd.GeoDataFrame.from_file(self.map_file)
                if area_map.crs is None:
                    area_map.crs = 'epsg:4326'

                geometry = [Point(xy) for xy in zip(stay_data3['lon'], stay_data3['lat'])]
                stay_data4 = gpd.GeoDataFrame(stay_data3, crs="epsg:4326", geometry=geometry)
                stay_data4.crs = 'epsg:4326'
                stay_data5 = gpd.sjoin(stay_data4, area_map, how='left', predicate='intersects', lsuffix='left',
                                       rsuffix='right')
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

        # assign holidays

        self.kept_data['holiday'] = 0
        self.kept_data.loc[(self.kept_data['weekday'] == 6) | (self.kept_data['weekday'] == 7), 'holiday'] = 1
        japan_holiday = japan_holidays.HolidayDataset.HOLIDAYS.keys()
        self.kept_data.loc[self.kept_data.arrival_time.dt.date.isin(japan_holiday), 'holiday'] = 1

        def _dbscan_individual(df):

            df = df.reset_index(drop=True)
            row_id = df.index.tolist()
            df['row_id'] = pd.DataFrame(row_id)[0]

            #  DBSCAN cluster for all records ##################################################################

            all_df = df.copy()
            all_df_point = all_df[["lon", "lat"]]
            all_df_for_dbsc = all_df_point.to_numpy(
                copy=True)  # convert to array   df2.as_matrix().astype("float64", copy=False)
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
                ((home_df1['hour'] >= 20) & (home_df1['end_hour'] >= 23)) | (home_df1['hour'] <= 4)]  # ??????home

            home_df4 = home_df3[home_df3['time_period_second'] >= 5400]  # ??????home
            if home_df4.empty is False:

                home_candidate = home_df4.reset_index(drop=True)
                home_candidate_point = home_candidate[["lon", "lat"]]
                home_candidate_for_dbsc = home_candidate_point.to_numpy(
                    copy=True)  # convert to array   df2.as_matrix().astype("float64", copy=False)
                home_candidate_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371),
                                             min_samples=self.dbscan_min_samples,
                                             algorithm='ball_tree', metric='haversine').fit(
                    np.radians(home_candidate_for_dbsc))
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
            work_df4 = work_df2[work_df2['time_period_second'] >= 5400]  # ??????work
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

            final_candidate.loc[:, ['home_label', 'work_label']] = final_candidate[['home_label', 'work_label']].fillna(
                -1)

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
                    home_counts1 = pd.value_counts(home_counts_df2['home_label']).sort_values(
                        ascending=False).to_frame()
                    home_counts1['home_index'] = home_counts1.index.tolist()
                    home_counts1.columns = ['num_home', 'home_index']
                    home_counts4 = home_counts1.reset_index(drop=True)
                    work_counts_df = one_cluster_df[one_cluster_df['work_label'] != -1]
                    work_counts_df2 = work_counts_df.drop_duplicates(['day'], keep='first')
                    work_counts1 = pd.value_counts(work_counts_df2['work_label']).sort_values(
                        ascending=False).to_frame()
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

            # determine home/work label order

            new_result_home = new_result[new_result['home_label_new'] != -1]
            if len(new_result_home) != 0:

                new_list = []

                home_new = new_result[new_result['home_label_new'] != -1]
                work_new = new_result[new_result['work_label_new'] != -1]
                other_new = new_result[new_result['other_label_new'] != -1]
                noise_new = new_result[new_result['all_detect_label'] == -1]

                if len(home_new) != 0:
                    home_order_counts = pd.value_counts(home_new['home_label_new']).sort_values(
                        ascending=False).to_frame()
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
                        ['user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon', 'hour', 'end_hour',
                         'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id', 'all_detect_label',
                         'home_label', 'work_label', 'home_label_new', 'work_label_new', 'other_label_new',
                         'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon',
                         'work_lat', 'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(home_group_0)

                if len(work_new) != 0:
                    work_order_counts = pd.value_counts(work_new['work_label_new']).sort_values(
                        ascending=False).to_frame()
                    work_order_counts['original_work_label'] = work_order_counts.index.tolist()
                    work_order_counts['work_label_new_order'] = list(range(len(work_order_counts)))
                    work_order_counts.columns = ['counts', 'original_work_label', 'work_label_order']

                    work_group = pd.merge(work_new, work_order_counts, left_on='work_label_new',
                                          right_on="original_work_label", how='left')

                    work_group['home_label_order'] = -1
                    work_group['original_home_label'] = -1
                    work_group['other_label_order'] = -1
                    work_group['original_other_label'] = -1

                    work_group_0 = work_group[
                        ['user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon', 'hour', 'end_hour',
                         'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id', 'all_detect_label',
                         'home_label', 'work_label', 'home_label_new', 'work_label_new', 'other_label_new',
                         'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon',
                         'work_lat', 'work_lon', 'other_lat', 'other_lon']]

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

                    other_group_0 = other_group[
                        ['user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon', 'hour', 'end_hour',
                         'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id', 'all_detect_label',
                         'home_label', 'work_label', 'home_label_new', 'work_label_new', 'other_label_new',
                         'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon',
                         'work_lat', 'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(other_group_0)

                if len(noise_new) != 0:
                    noise_group = noise_new.copy()
                    noise_group['work_label_order'] = -1
                    noise_group['original_work_label'] = -1
                    noise_group['other_label_order'] = -1
                    noise_group['original_other_label'] = -1
                    noise_group['home_label_order'] = -1
                    noise_group['original_home_label'] = -1
                    noise_group_0 = noise_group[
                        ['user_id', 'day', 'arrival_time', 'departure_time', 'lat', 'lon', 'hour', 'end_hour',
                         'weekday', 'holiday', 'time_period', 'time_period_second', 'row_id', 'all_detect_label',
                         'home_label', 'work_label', 'home_label_new', 'work_label_new', 'other_label_new',
                         'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon',
                         'work_lat', 'work_lon', 'other_lat', 'other_lon']]

                    new_list.append(noise_group_0)

                _final_result = pd.concat(new_list, axis=0)

                final_result_home = _final_result[_final_result['home_label_order'] != -1]

                if self.map_file is not None:
                    great_tokyo_map = gpd.GeoDataFrame.from_file(self.map_file)
                    great_tokyo_map.crs = 'epsg:4326'

                    geometry = [Point(xy) for xy in zip(final_result_home['home_lon'], final_result_home['home_lat'])]
                    df_final_home = gpd.GeoDataFrame(final_result_home, crs="epsg:4326", geometry=geometry)
                    df_final_home.crs = 'epsg:4326'
                    df_whether_within = gpd.sjoin(df_final_home, great_tokyo_map, how='left', predicate='intersects',
                                                  lsuffix='left',
                                                  rsuffix='right')
                    df_whether_within['within_right'].fillna(-1)
                    df_final_within = df_whether_within[df_whether_within['within_right'] == 1]
                else:
                    df_final_within = final_result_home

                if len(df_final_within) != 0:
                    final_result2 = _final_result.sort_values(by='arrival_time')
                    return final_result2
            return

        final_result = apply_parallel(self.kept_data.groupby('user_id'), _dbscan_individual)
        self.home_work_result = final_result
        return final_result

    def extract_life_pattern(self):

        def _extract_life_pattern(df0):
            df_list = []

            for key_day, item_day in df0.groupby(by=['day']):
                date = key_day
                df0 = item_day.copy()

                df00 = df0.sort_values('time_period_second').drop_duplicates(['hour'], keep='last')
                df00['arrival_time'] = pd.to_datetime(df00['arrival_time'])
                _df = df00.sort_values(by='arrival_time')

                week_day = _df['weekday'].mean()
                holiday = _df['holiday'].mean()

                df_date = pd.DataFrame({'time': list(range(0, 24)), 'places': [-1] * 24})
                for t, row in _df.iterrows():
                    s_time = row['hour']
                    e_time = row['end_hour']
                    if (row['home_label_order'] == -1) & (row['work_label_order'] == -1) & (
                            row['other_label_order'] == -1):
                        pass
                    else:
                        if (row['home_label_order'] != -1) & (row['work_label_order'] == -1) & (
                                row['other_label_order'] == -1):
                            home_label = 'H' + '_' + str(int(row['home_label_order']))
                            df_date.loc[df_date['time'] == s_time, 'places'] = home_label
                            df_date.loc[df_date['time'] == e_time, 'places'] = home_label
                        elif (row['home_label_order'] == -1) & (row['work_label_order'] != -1) & (
                                row['other_label_order'] == -1):
                            work_label = 'W' + '_' + str(int(row['work_label_order']))
                            df_date.loc[df_date['time'] == s_time, 'places'] = work_label
                            df_date.loc[df_date['time'] == e_time, 'places'] = work_label
                        elif (row['home_label_order'] == -1) & (row['work_label_order'] == -1) & (
                                row['other_label_order'] != -1):
                            other_label = 'O' + '_' + str(int(row['other_label_order']))
                            df_date.loc[df_date['time'] == s_time, 'places'] = other_label
                            df_date.loc[df_date['time'] == e_time, 'places'] = other_label

                places_index = list(df_date[df_date['places'] != -1].index)
                if len(places_index) != 0:
                    places_split = np.split(df_date, places_index, axis=0)
                    places_list = []
                    for j in range(0, len(places_split)):
                        if j == 0:
                            places_df = places_split[j].copy()

                            home_order = _df[_df['home_label_order'] != -1]
                            if len(home_order) != 0:
                                home_num = home_order['home_label_order'].min()
                                places_df['places'] = 'H' + '_' + str(home_num)
                            if len(home_order) == 0:
                                places_df['places'] = "H_0"
                            places_list.append(places_df)
                        elif j != 0:
                            places_df = places_split[j].copy()
                            places_df = places_df.reset_index(drop=True)
                            places_df['places'] = places_df.loc[0, 'places']
                            places_list.append(places_df)

                    places_concat = pd.concat(places_list)
                    places_concat = places_concat.reset_index(drop=True)
                    places_concat['next_places'] = places_concat['places'].shift(-1)
                    places_concat.loc[23, 'next_places'] = places_concat.loc[23, 'places']
                    places_concat['user_id'] = df0.user_id.values[0]
                    places_concat['date'] = date
                    places_concat['week_day'] = week_day
                    places_concat['holiday'] = holiday

                    df_list.append(places_concat)

            if len(df_list) != 0:
                final_pattern_df = pd.concat(df_list)
                return final_pattern_df
            else:
                return

        df = apply_parallel(self.home_work_result.groupby('user_id'), _extract_life_pattern)
        self.life_pattern = df
        return df

    def support_tree(self):

        # concat all the possible node of the tree and drop duplicate node.
        def _support_tree(df):
            list_1w = []
            list_20w = []
            list_total = []

            try:

                list_1w.append(df)
                concat_1w = pd.concat(list_1w)
                _concat_1W_2 = concat_1w.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
                list_20w.append(_concat_1W_2)
                list_1w = []

                concat_20w = pd.concat(list_20w)
                concat_20w_2 = concat_20w.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
                list_total.append(concat_20w_2)
                list_20w = []

            except OSError:
                print('ERROR.')

            if len(list_1w) != 0:
                concat_1w_rest = pd.concat(list_1w)
                _concat_1W_rest_2 = concat_1w_rest.drop_duplicates(subset=['time', 'places', 'next_places'],
                                                                   keep='first')
                list_total.append(_concat_1W_rest_2)

            if len(list_20w) != 0:
                concat_20w_rest = pd.concat(list_20w)
                concat_20w_rest_2 = concat_20w_rest.drop_duplicates(subset=['time', 'places', 'next_places'],
                                                                    keep='first')
                list_total.append(concat_20w_rest_2)

            _tree_concat = pd.concat(list_total)
            tree_concat2 = _tree_concat.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
            tree_concat3 = tree_concat2.sort_values(by=['time', 'places', 'next_places'], ascending=True)

            # give each node a  unique index
            tree_concat3['tree_index'] = list(range(0, len(tree_concat3)))
            return tree_concat3

        tree_concat = apply_parallel(self.life_pattern.groupby('user_id'), _support_tree)
        self.tree_concat = tree_concat
        return tree_concat

    def merge_tree(self, save_support_tree=False):

        list_df = []
        for key, item in self.tree_concat.groupby(by='time'):
            df_one = item
            df_one2 = df_one.drop_duplicates(subset=['places', 'next_places'], keep='first')
            list_df.append(df_one2)

        df_final = pd.concat(list_df, axis=0)

        df_final2 = df_final.sort_values(by=['time', 'places', 'next_places'], ascending=True)
        df_final2['tree_index'] = list(range(0, len(df_final2)))
        if save_support_tree:

            df_final2.to_csv(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv', sep=',',
                             encoding='utf-8')
        self.merged_tree = df_final2

        return df_final2

    def pattern_probability_matrix(self, using_exit_tree=''):

        if os.path.exists(using_exit_tree):
            df_total_tree_index = pd.read_csv(using_exit_tree)

        else:
            print('No existing support tree...\nGenerating support tree...')
            df_total_tree_index = self.merged_tree

        def _pattern_probability_matrix(df):
            user_list = []
            for key_day, item_day in df.groupby(by='date'):
                df_1u_1day = item_day

                df_one_user2 = df_1u_1day[['time', 'places', 'next_places']].copy()
                df_one_user2.loc[:, 'attribute'] = 1

                df_tree_total2 = df_total_tree_index[['tree_index', 'time', 'places', 'next_places']].copy()
                df_user_matrix = pd.merge(df_tree_total2, df_one_user2, how='left')

                df_user_matrix2 = df_user_matrix.fillna(0)
                df_user_matrix3 = df_user_matrix2[['tree_index', 'attribute']].copy()
                user_list.append(df_user_matrix3['attribute'])

            df_user_concat = pd.concat(user_list, axis=1)
            aaa = df_user_concat.mean(1)

            return pd.DataFrame(aaa).T

        _pattern_probability = apply_parallel(self.tree_concat.groupby('user_id'), _pattern_probability_matrix)
        self.pattern_probability_mat = _pattern_probability
        return _pattern_probability

    def nmf_average(self, save_results=True, save_visualization=True):

        data = self.pattern_probability_mat
        # user_matrix_list = np.array(data).flatten('F')
        user_matrix_list = np.array(data)
        user_id_list = self.tree_concat.user_id.unique()

        # non-negative matrix factorization
        user_matrix1 = np.array(user_matrix_list)
        user_matrix = np.transpose(user_matrix1)

        model = NMF(n_components=3, init='nndsvda')

        w = model.fit_transform(user_matrix)

        h = model.components_

        df_w = pd.DataFrame(w)
        df_h = pd.DataFrame(h)

        # filename_df = pd.DataFrame(filename_list)
        filename_df = pd.DataFrame(user_id_list)

        if save_results:

            df_w.to_csv(self.NMF_results_folder + 'W_multiple_HW_single_O_total_day.csv', index=False)
            df_h.to_csv(self.NMF_results_folder + 'H_multiple_HW_single_O_total_day.csv', index=False)
            filename_df.to_csv(self.NMF_results_folder + 'filename_multiple_HW_single_O_total_day.csv', index=False)

        # visualization
        if save_visualization:
            df_h_ = df_h.T

            df_h_['user_id'] = user_id_list
            df_h_.columns = ['x', 'y', 'z', 'user_id']

            fig = plt.figure(figsize=(8, 5), dpi=300)
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            ax.scatter(df_h_['x'].values, df_h_['y'].values, df_h_['z'].values, s=20, marker='o')
            ax.view_init(elev=35, azim=45)
            plt.savefig(self.NMF_results_folder + 'demo_result.png')
            plt.close()

        return df_w, df_h, filename_df

    # def plus_home_work_location(self, save_results=False, new_gps_data=None):
    #
    #     if self.initialize:
    #         df_clustering_result = self.clustering()
    #     else:
    #         if os.path.exists(self.clustering_results_folder + 'demo_2_KMeans.csv'):
    #             df_clustering_result = pd.read_csv(self.clustering_results_folder + 'demo_2_KMeans.csv')
    #         else:
    #             df_clustering_result = self.clustering(new_gps_data=new_gps_data)
    #
    #     for i in range(len(df_clustering_result)):
    #
    #         # print('processing:', i)
    #
    #         user_ID = str(df_clustering_result.loc[i, 'user_id']).zfill(8)
    #
    #         # if os.path.exists('./')
    #
    #         # df_home_work = pd.read_csv(home_work_location_dir + '//' + user_ID + '_great_tokyo_labeled.csv')
    #         self.raw_gps_file = self.raw_gps_folder + user_ID + '.csv'
    #         # self.raw_gps_file = user_ID
    #         # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
    #         df_home_work = self.detect_home_work(raw_gps_file=self.raw_gps_file)
    #
    #         df_home = df_home_work[df_home_work['home_label_order'] == 0]
    #
    #         df_work = df_home_work[df_home_work['work_label_order'] == 0]
    #
    #         if len(df_home) != 0:
    #             home_lng = df_home['home_lon'].mean()
    #             home_lat = df_home['home_lat'].mean()
    #
    #         if len(df_home) == 0:
    #             home_lng = -1
    #             home_lat = -1
    #
    #         if len(df_work) != 0:
    #             work_lng = df_work['work_lon'].mean()
    #             work_lat = df_work['work_lat'].mean()
    #
    #         if len(df_work) == 0:
    #             work_lng = -1
    #             work_lat = -1
    #
    #         df_clustering_result.loc[i, 'home_lat'] = home_lat
    #         df_clustering_result.loc[i, 'home_lng'] = home_lng
    #         df_clustering_result.loc[i, 'work_lat'] = work_lat
    #         df_clustering_result.loc[i, 'work_lng'] = work_lng
    #
    #     if save_results:
    #         if self.initialize:
    #             df_clustering_result.to_csv(self.clustering_results_folder + '8_ID_XYZ_groupID_homeworklocation.csv')
    #         else:
    #             df_clustering_result.to_csv(
    #             self.clustering_results_folder + 'New_ID_XYZ_groupID_homeworklocation.csv')
    #
    #     # print('finish')
    #
    # def csv2Point(self, data, LON, LAT, coordinate):
    #
    #     data_point = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[LON], data[LAT]), crs=coordinate)
    #
    #     return data_point
    #
    # def Point2Name(self, SHP_PATH, POINT, COLUMNS, coordinate):
    #
    #     chome = gpd.read_file(SHP_PATH).to_crs({'init': coordinate})
    #
    #     COLUMNS.append('geometry')
    #     chome = chome[COLUMNS]
    #     join_point = gpd.sjoin(POINT, chome, how='left', op='intersects', lsuffix='left', rsuffix='right')
    #     join_point = join_point.drop(['index_right', 'geometry'], axis=1)
    #     return join_point
    #
    # def generate_key_point_table(self, save_results=False, new_gps_data=None):
    #
    #     if os.path.exists(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv'):
    #         df_totaltree = pd.read_csv(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv')
    #     else:
    #         df_totaltree = self.merge_tree()
    #     # df_totaltree = pd.read_csv(tree_index_dir + '//the_great_tokyo_total_tree_index.csv')
    #     significant_place_list0 = df_totaltree['places'].tolist() + df_totaltree['next_places'].tolist()
    #     significant_place_list = list(set(significant_place_list0))
    #     significant_place_df = pd.DataFrame(significant_place_list, columns=['place_name'])
    #     a = significant_place_df['place_name'].str.split("_", expand=True)
    #     a.columns = ['place', 'order']
    #     a.loc[:, 'order'] = a['order'].fillna(0)
    #     a['order'] = a['order'].astype('int')
    #     a = a.sort_values(['place', 'order'], ascending=[True, True])
    #     b = pd.concat([a[a['place'] == 'H'], a[a['place'] == 'W'], a[a['place'] == 'O']])
    #     b = b.reset_index(drop=True)
    #
    #     dic_ = {"H": 'home', "W": 'work', 'O': "other"}
    #
    #     # 2.find key point
    #     chome = gpd.read_file('./gis/h12ka13.shp')
    #     chome = chome[['KEY_CODE', 'PREF_NAME', 'CITY_NAME', 'S_NAME', 'geometry']]
    #     if self.initialize:
    #         if os.path.exists(self.clustering_results_folder + 'demo_2_KMeans.csv'):
    #             df_clustering_result = pd.read_csv(self.clustering_results_folder + 'demo_2_KMeans.csv')
    #         else:
    #             df_clustering_result = self.clustering()
    #     else:
    #         df_clustering_result = self.clustering(new_gps_data=new_gps_data)
    #
    #     for i in range(len(df_clustering_result)):
    #
    #         # print('processing:', i)
    #
    #         user_ID = str(df_clustering_result.loc[i, 'user_id']).zfill(8)
    #
    #         # df_home_work = pd.read_csv(home_work_location_dir + '//' + user_ID + '_great_tokyo_labeled.csv')
    #         # self.raw_gps_file = user_ID
    #         self.raw_gps_file = self.raw_gps_folder + user_ID + '.csv'
    #         # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
    #         df_home_work = self.detect_home_work(raw_gps_file=self.raw_gps_file)
    #
    #         individual_key_point_df = b.copy()
    #         individual_key_point_df['user_ID'] = user_ID
    #
    #         for k in range(len(individual_key_point_df)):
    #             place = individual_key_point_df.loc[k, 'place']
    #             order = individual_key_point_df.loc[k, 'order']
    #
    #             individual_key_point_df.loc[k, 'significant_place'] = place + '_' + str(order)
    #             temp = df_home_work[df_home_work[dic_[place] + '_label_order'] == order]
    #
    #             if len(temp) != 0:
    #                 Lng = temp[dic_[place] + '_lon'].mean()
    #                 Lat = temp[dic_[place] + '_lat'].mean()
    #                 mesh_code = ju.to_meshcode(Lat, Lng, 5)
    #
    #                 individual_key_point_df.loc[k, 'Lng'] = Lng
    #                 individual_key_point_df.loc[k, 'Lat'] = Lat
    #                 individual_key_point_df.loc[k, 'mesh_code'] = mesh_code
    #
    #         individual_key_point_df = individual_key_point_df.fillna(-1)
    #
    #         point = self.csv2Point(individual_key_point_df, 'Lng', 'Lat', 'EPSG:4326')
    #         join_point = self.Point2Name('./gis/h12ka13.shp', point, ['KEY_CODE', 'PREF_NAME', 'CITY_NAME', 'S_NAME'],
    #                                      'EPSG:4326')
    #         join_point = join_point.fillna(-1)
    #         if save_results:
    #             if not os.path.exists('./key_point_table/'):
    #                 os.mkdir('./key_point_table/')
    #             join_point.to_csv('./key_point_table/' + str(self.user_id) + '_key_point_table.csv', index=False)
    #     return join_point
    #
    # def generate_group_HWO_join_area2(self, save_results=False, new_gps_data=None):
    #     if self.initialize:
    #         if os.path.exists(self.clustering_results_folder + 'demo_2_KMeans.csv'):
    #             df = pd.read_csv(self.clustering_results_folder + 'demo_2_KMeans.csv')
    #         else:
    #             df = self.clustering()
    #     else:
    #         df = self.clustering(new_gps_data=new_gps_data)
    #     # ??????df = pd.read_csv('9_conduct_same_cluster_as2013//2011_7_cluster_KMeans.csv')
    #
    #     for i in range(len(df)):
    #         user_ID = str(df.loc[i, 'user_id']).zfill(8)
    #         # df_1_user_stay = pd.read_csv(
    #         #     '2_great_tokyo_detect_home_work_order' + '//' + userID + '_great_tokyo_labeled.csv')
    #         # self.raw_gps_file = user_ID
    #         self.raw_gps_file = self.raw_gps_folder + user_ID + '.csv'
    #         # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
    #         df_1_user_stay = self.detect_home_work(self.raw_gps_file)
    #         df.loc[i, 'home_Lat_WGS84'] = df_1_user_stay[df_1_user_stay['home_label_order'] == 0]['home_lat'].mean()
    #         df.loc[i, 'home_Lng_WGS84'] = df_1_user_stay[df_1_user_stay['home_label_order'] == 0]['home_lon'].mean()
    #         df.loc[i, 'work_Lat_WGS84'] = df_1_user_stay[df_1_user_stay['work_label_order'] == 0]['work_lat'].mean()
    #         df.loc[i, 'work_Lng_WGS84'] = df_1_user_stay[df_1_user_stay['work_label_order'] == 0]['work_lon'].mean()
    #
    #     df['home_Lat_WGS84'].fillna(-1, inplace=True)
    #     df['home_Lng_WGS84'].fillna(-1, inplace=True)
    #     df_point = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['home_Lng_WGS84'], df['home_Lat_WGS84']),
    #                                 crs='EPSG:4326')
    #     Tokyo_map = gpd.read_file('./gis/tokyo_area_statistic.shp').to_crs({'init': 'EPSG:4326'})
    #     Tokyo_map = Tokyo_map[['N03_001', 'N03_003', 'N03_004', 'geometry']]
    #     join_point = gpd.sjoin(df_point, Tokyo_map, how='left', op='intersects', lsuffix='left', rsuffix='right')
    #     join_point = join_point.drop(['index_right', 'geometry'], axis=1)
    #     join_point = join_point.fillna(-1)
    #     if save_results:
    #         join_point.to_csv('./11_demo_7_group_HWO_join_area2.csv', index=False)


def apply_parallel(df_grouped, func):
    ret_lst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in df_grouped)
    return pd.concat(ret_lst)


if __name__ == '__main__':
    lpp_v2 = LifePatternProcessor()
    stay_data2 = lpp_v2.select_area(
        raw_gps_file='../functions/stay_point_detection/dataset_TSMC2014_TKY_stay_points.csv',
        map_file=None)
    container = lpp_v2.detect_home_work()
    sample = lpp_v2.extract_life_pattern()
    support_tree = lpp_v2.support_tree()
    merged_tree = lpp_v2.merge_tree(save_support_tree=False)
    pattern_probability = lpp_v2.pattern_probability_matrix()
    print(np.shape(pattern_probability))
    _, _, _ = lpp_v2.nmf_average()
    print('here')
