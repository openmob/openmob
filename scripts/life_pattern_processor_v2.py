import glob
import os.path
from sklearn.decomposition import NMF
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import DBSCAN
import sklearn.cluster
import joblib
from scipy.optimize import least_squares
from tqdm import tqdm
import jismesh.utils as ju
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')
matplotlib.use('Agg')


class LifePatternProcessor:

    def __init__(self,
                 raw_gps_folder='../../datasets/separate_trip/',
                 map_file='../../datasets/gis/greater_tokyo_area_dissolve.shp',
                 distance_for_eps=0.03,
                 dbscan_min_samples=10,
                 initialize=False,
                 support_tree_folder='./support_tree/',
                 NMF_results_folder='./NMF_results/',
                 clustering_results_folder='./clustering_results/'):

        self.user_id_8 = None
        self.user_id = None
        self.raw_gps_folder = raw_gps_folder
        self.distance_for_eps = distance_for_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.raw_gps_file = None
        self.map_file = map_file
        self.support_tree_folder = support_tree_folder
        self.NMF_results_folder = NMF_results_folder
        self.clustering_results_folder = clustering_results_folder
        self.initialize = initialize

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

    def select_great_tokyo(self, raw_gps_file):
        if self.raw_gps_file is None:
            self.raw_gps_file = raw_gps_file
        great_tokyo_map = gpd.GeoDataFrame.from_file(self.map_file)
        if great_tokyo_map.crs is None:
            great_tokyo_map.crs = 'epsg:4326'
        data = pd.read_csv(self.raw_gps_file, header=None)
        self.user_id = data.iloc[0, 0]
        self.user_id_8 = str(self.user_id).zfill(8)
        if pd.isnull(self.user_id):
            print('user_id is null...')
            return
        else:
            stay_data = data[[0, 1, 3, 4, 10, 11, 16]][(data[3] == 'STAY')]
            if stay_data.empty is True:
                print('stay data is empty...')
                return
            elif stay_data.empty is False:
                spl_data = stay_data[16].str.split("|", expand=True)
                stay_data2 = pd.concat([stay_data, spl_data], axis=1)
                stay_data2.columns = ['user_id', 'day', 'mode', 'mode2', 'start_time', 'end_time', 'trajectory',
                                      'number',
                                      'time', 'latitude', 'longitude']
                stay_data2 = stay_data2.drop(['trajectory', 'number'], axis=1)
                stay_data2['start_time'] = pd.to_datetime(stay_data2['start_time'])
                stay_data20 = stay_data2.sort_values(by='start_time')
                stay_data20['longitude'] = stay_data20['longitude'].astype(float)
                stay_data20['latitude'] = stay_data20['latitude'].astype(float)
                stay_data3 = stay_data20[
                    (stay_data20['longitude'] >= 125) & (stay_data20['latitude'] >= 20)]  # 不在日本境内的被剔除
                stay_data3 = stay_data3.reset_index(drop=True)
                stay_data3['hour'] = pd.to_datetime(stay_data3['time']).dt.hour
                stay_data3['weekday'] = pd.to_datetime(stay_data3['time']).dt.weekday
                stay_data3['weekday'] = stay_data3['weekday'] + 1
                stay_data3['time_period'] = pd.to_datetime(stay_data3['end_time']) - pd.to_datetime(
                    stay_data3['start_time'])
                stay_data3['time_period_second'] = (
                        pd.to_datetime(stay_data3['end_time']) - pd.to_datetime(stay_data3['start_time'])).dt.seconds
                geometry = [Point(xy) for xy in zip(stay_data3['longitude'], stay_data3['latitude'])]
                stay_data4 = gpd.GeoDataFrame(stay_data3, crs="epsg:4326", geometry=geometry)
                stay_data4.crs = 'epsg:4326'
                stay_data5 = gpd.sjoin(stay_data4, great_tokyo_map, how='left', predicate='intersects', lsuffix='left',
                                       rsuffix='right')
                stay_data6 = stay_data5.reset_index(drop=True)
                stay_data6 = stay_data6.drop(['geometry', 'index_right'], axis=1)
                stay_data6['within'] = stay_data6['within'].fillna(-1)
                tokyo_stay = stay_data6[stay_data6['within'] == 1]

                if len(stay_data6) != 0:
                    if (len(stay_data6) > 10) & (len(tokyo_stay) >= 4) & ((len(tokyo_stay) / len(stay_data6)) >= 0.2):
                        len_st6 = len(stay_data6)
                        len_tks = len(tokyo_stay)
                        # print(self.user_id_8, len_st6, len_tks)
                        return stay_data6
                    else:
                        len_st6 = -1
                        len_tks = 0
                        # print(self.user_id_8, len_st6, len_tks)
                        return
                else:
                    len_st6 = -1
                    len_tks = 0
                    # print(self.user_id_8, len_st6, len_tks)
                    return

    def detect_home_work(self, raw_gps_file):

        if self.raw_gps_file is None:
            self.raw_gps_file = raw_gps_file
        df0 = self.select_great_tokyo(self.raw_gps_file)
        if df0 is None:
            print('No stay data is given...')
            return

        df000 = df0.copy(deep=False)

        df000['start_time'] = pd.to_datetime(df0['start_time'])
        df = df000.sort_values(by='start_time')
        df = df.reset_index(drop=True)
        row_id = df.index.tolist()
        df['row_ID'] = pd.DataFrame(row_id)[0]
        df['end_hour'] = pd.to_datetime(df['end_time']).dt.hour
        df['holiday'] = 0
        df.loc[(df['weekday'] == 6) | (df['weekday'] == 7), 'holiday'] = 1
        df.loc[(df['day'] == '2013-01-01') | (df['day'] == '2013-01-14') | (df['day'] == '2013-02-11') | (
                df['day'] == '2013-03-20') | (df['day'] == '2013-04-29') | (df['day'] == '2013-05-03') | (
                       df['day'] == '2013-05-04') | (df['day'] == '2013-05-05') | (df['day'] == '2013-07-15') | (
                       df['day'] == '2013-09-16') | (df['day'] == '2013-09-23') | (df['day'] == '2013-10-14') | (
                       df['day'] == '2013-11-03') | (df['day'] == '2013-11-23') | (
                       df['day'] == '2013-12-23'), 'holiday'] = 1

        #  DBSCAN cluster for all records ##################################################################

        all_df = df.copy()
        all_df_point = all_df[["longitude", "latitude"]]
        all_df_for_dbsc = all_df_point.to_numpy( copy=True)  # convert to array   df2.as_matrix().astype("float64", copy=False)
        all_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371), min_samples=self.dbscan_min_samples, algorithm='ball_tree',
                          metric='haversine').fit(np.radians(all_df_for_dbsc))
        all_labels = all_dbsc.labels_
        all_n_clusters_ = len(set(all_labels)) - (1 if -1 in all_labels else 0)
        all_total_n_noise_ = list(all_labels).count(-1)
        all_labels_list = all_labels.tolist()
        all_df['all_detect_label'] = pd.DataFrame(all_labels_list)

        # DBSCAN cluster for HOME candidates

        home_df = df.copy()
        home_df1 = home_df[home_df['holiday'] != 1]
        home_df3 = home_df1[
            ((home_df1['hour'] >= 20) & (home_df1['end_hour'] >= 23)) | (home_df1['hour'] <= 4)]  # 粗选home
        # home_df3 = home_df1[(home_df1['hour'] >= 20)  | (home_df1['hour'] <= 4)]   # for order
        home_df4 = home_df3[home_df3['time_period_second'] >= 5400]  # 粗选home
        if home_df4.empty is False:

            home_candidate = home_df4.reset_index(drop=True)
            home_candidate_point = home_candidate[["longitude", "latitude"]]
            home_candidate_for_dbsc = home_candidate_point.to_numpy(
                copy=True)  # convert to array   df2.as_matrix().astype("float64", copy=False)
            home_candidate_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371), min_samples=self.dbscan_min_samples,
                                         algorithm='ball_tree', metric='haversine').fit(
                np.radians(home_candidate_for_dbsc))
            home_candidate_labels = home_candidate_dbsc.labels_
            home_candidate_n_clusters_ = len(set(home_candidate_labels)) - (1 if -1 in home_candidate_labels else 0)
            home_candidate_total_n_noise_ = list(home_candidate_labels).count(-1)
            home_candidate_labels_list = home_candidate_labels.tolist()
            home_candidate['home_label'] = pd.DataFrame(home_candidate_labels_list)
            home_candidate2 = home_candidate[['row_ID', 'home_label']]
            home_df5 = pd.merge(all_df, home_candidate2, left_on='row_ID', right_on='row_ID', how='left')
        else:
            home_df5 = all_df.copy()
            home_df5['home_label'] = -1

        # DBSCAN cluster for WORK candidates
        work_df = df.copy()
        work_day_df_list = []

        work_df1 = work_df[work_df['holiday'] != 1]
        work_df2 = work_df1[(work_df1['end_hour'] <= 19) & (work_df1['hour'] >= 8)]
        work_df4 = work_df2[work_df2['time_period_second'] >= 5400]  # 粗选work
        if work_df4.empty is False:

            work_candidate = work_df4.reset_index(drop=True)
            work_candidate_point = work_candidate[["longitude", "latitude"]]
            work_candidate_for_dbsc = work_candidate_point.to_numpy(copy=True)
            work_candidate_dbsc = DBSCAN(eps=(self.distance_for_eps / 6371), min_samples=self.dbscan_min_samples,
                                         algorithm='ball_tree',
                                         metric='haversine').fit(np.radians(work_candidate_for_dbsc))
            work_candidate_labels = work_candidate_dbsc.labels_
            work_candidate_n_clusters_ = len(set(work_candidate_labels)) - (1 if -1 in work_candidate_labels else 0)
            work_candidate_total_n_noise_ = list(work_candidate_labels).count(-1)
            work_candidate_labels_list = work_candidate_labels.tolist()
            work_candidate['work_label'] = pd.DataFrame(work_candidate_labels_list)
            # work_candidate.to_csv(csv_labeled_dir + i + '_labeled_work.csv', index=None)
            work_candidate2 = work_candidate[['row_ID', 'work_label']]
            final_candidate = pd.merge(home_df5, work_candidate2, left_on='row_ID', right_on='row_ID', how='left')
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
                day_num_of_cluster = len(one_cluster_df.groupby(by=['day']))
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
                        one_cluster_df2['home_lat'] = one_cluster_df2['latitude'].mean()
                        one_cluster_df2['home_lon'] = one_cluster_df2['longitude'].mean()
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
                        one_cluster_df2['work_lat'] = one_cluster_df2['latitude'].mean()
                        one_cluster_df2['work_lon'] = one_cluster_df2['longitude'].mean()
                        one_cluster_df2['other_lat'] = -1
                        one_cluster_df2['other_lon'] = -1
                        one_cluster_list.append(one_cluster_df2)

                elif (len(home_counts4) != 0) & (len(work_counts4) == 0):
                    home_max_number = home_counts4.loc[0, 'num_home']
                    home_max_index = home_counts4.loc[0, 'home_index']
                    one_cluster_df2 = one_cluster_df
                    one_cluster_df2['home_label_new'] = home_max_index
                    one_cluster_df2['work_label_new'] = -1
                    one_cluster_df2['other_label_new'] = -1
                    one_cluster_df2['home_lat'] = one_cluster_df2['latitude'].mean()
                    one_cluster_df2['home_lon'] = one_cluster_df2['longitude'].mean()
                    one_cluster_df2['work_lat'] = -1
                    one_cluster_df2['work_lon'] = -1
                    one_cluster_df2['other_lat'] = -1
                    one_cluster_df2['other_lon'] = -1
                    one_cluster_list.append(one_cluster_df2)

                elif (len(home_counts4) == 0) & (len(work_counts4) != 0):
                    work_max_number = work_counts4.loc[0, 'num_work']
                    work_max_index = work_counts4.loc[0, 'work_index']
                    one_cluster_df2 = one_cluster_df
                    one_cluster_df2['home_label_new'] = -1
                    one_cluster_df2['work_label_new'] = work_max_index
                    one_cluster_df2['other_label_new'] = -1
                    one_cluster_df2['home_lat'] = -1
                    one_cluster_df2['home_lon'] = -1
                    one_cluster_df2['work_lat'] = one_cluster_df2['latitude'].mean()
                    one_cluster_df2['work_lon'] = one_cluster_df2['longitude'].mean()
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
                    one_cluster_df2['other_lat'] = one_cluster_df2['latitude'].mean()
                    one_cluster_df2['other_lon'] = one_cluster_df2['longitude'].mean()
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
                    ['user_id', 'day', 'mode', 'mode2', 'start_time', 'end_time', 'time', 'latitude', 'longitude',
                     'hour',
                     'end_hour', 'weekday', 'within', 'holiday', 'time_period', 'time_period_second', 'row_ID',
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
                    'user_id', 'day', 'mode', 'mode2', 'start_time', 'end_time', 'time', 'latitude', 'longitude',
                    'hour',
                    'end_hour', 'weekday', 'within', 'holiday', 'time_period', 'time_period_second', 'row_ID',
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
                    'user_id', 'day', 'mode', 'mode2', 'start_time', 'end_time', 'time', 'latitude', 'longitude',
                    'hour',
                    'end_hour', 'weekday', 'within', 'holiday', 'time_period', 'time_period_second', 'row_ID',
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
                    'user_id', 'day', 'mode', 'mode2', 'start_time', 'end_time', 'time', 'latitude', 'longitude',
                    'hour',
                    'end_hour', 'weekday', 'within', 'holiday', 'time_period', 'time_period_second', 'row_ID',
                    'all_detect_label', 'home_label', 'work_label', 'home_label_new', 'work_label_new',
                    'other_label_new',
                    'home_label_order', 'work_label_order', 'other_label_order', 'home_lat', 'home_lon', 'work_lat',
                    'work_lon', 'other_lat', 'other_lon']]

                new_list.append(noise_group_0)

            final_result = pd.concat(new_list, axis=0)

            final_result_home = final_result[final_result['home_label_order'] != -1]

            great_tokyo_map = gpd.GeoDataFrame.from_file('gis//greater_tokyo_area_dissolve.shp')
            great_tokyo_map.crs = 'epsg:4326'

            geometry = [Point(xy) for xy in zip(final_result_home['home_lon'], final_result_home['home_lat'])]
            df_final_home = gpd.GeoDataFrame(final_result_home, crs="epsg:4326", geometry=geometry)
            df_final_home.crs = 'epsg:4326'
            df_wether_within = gpd.sjoin(df_final_home, great_tokyo_map, how='left', predicate='intersects', lsuffix='left',
                                         rsuffix='right')
            df_wether_within['within_right'].fillna(-1)
            df_final_within = df_wether_within[df_wether_within['within_right'] == 1]

            if len(df_final_within) != 0:
                final_result2 = final_result.sort_values(by='start_time')
                return final_result2
        return

    def extract_life_pattern(self, raw_gps_file):

        if self.raw_gps_file is None:
            self.raw_gps_file = raw_gps_file
        df0 = self.detect_home_work(self.raw_gps_file)

        df_list = []

        for key_day, item_day in df0.groupby(by=['day']):  # 拆分每个用户每一天
            date = key_day
            df0 = item_day.copy()

            # 如果有同一小时内的记录，保留停留时间较长的那条
            df00 = df0.sort_values('time_period_second').drop_duplicates(['hour'],
                                                                         keep='last')  # ascending = True 为升序，默认
            df00['start_time'] = pd.to_datetime(df00['start_time'])
            df = df00.sort_values(by='start_time')

            week_day = df['weekday'].mean()
            holiday = df['holiday'].mean()

            # 从0点到23点 填充记录

            s_list = []
            e_list = []
            df_date = pd.DataFrame({'time': list(range(0, 24)), 'places': [-1] * 24})
            for t, row in df.iterrows():
                s_time = row['hour']
                e_time = row['end_hour']
                if (row['home_label_order'] == -1) & (row['work_label_order'] == -1) & (row['other_label_order'] == -1):
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

            # 补足没有记录的行
            places_index = list(df_date[df_date['places'] != -1].index)
            if len(places_index) != 0:
                places_split = np.split(df_date, places_index, axis=0)
                places_list = []
                for j in range(0, len(places_split)):
                    if j == 0:
                        places_df = places_split[j].copy()

                        home_order = df[df['home_label_order'] != -1]
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
                places_concat['ID'] = self.user_id
                places_concat['date'] = date
                places_concat['week_day'] = week_day
                places_concat['holiday'] = holiday

                df_list.append(places_concat)

        if len(df_list) != 0:
            final_pattern_df = pd.concat(df_list)
            return final_pattern_df

        return

    def support_tree(self):

        if self.raw_gps_folder is None:
            print('Loading sample ZDC GPS folder...')
            self.raw_gps_folder = '../sample/'

        filename_list = glob.glob(self.raw_gps_folder + '*.csv')

        # print('==============', len(filename_list), '==============')

        # concat all the posible node of the tree and drop duplicate node.
        error_list = []
        tree_df_list = []
        list_1w = []
        list_20w = []
        list_total = []
        j = 0
        m = 0
        for k in range(len(filename_list)):
            try:
                self.raw_gps_file = filename_list[k]
                # print(self.raw_gps_file)
                # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
                df = self.extract_life_pattern(self.raw_gps_file)
                list_1w.append(df)
                if k % 100 == 0 and k > 10:
                    concat_1w = pd.concat(list_1w)
                    concat_1W_2 = concat_1w.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
                    list_20w.append(concat_1W_2)
                    list_1w = []
                    j += 1
                    # print('==========', 'file:', k, '==========')
                    # print('==========', 'progress 1W:', j, 'progress 20w:', m, '==========')

                if k % 2000 == 0 and k > 100:
                    concat_20w = pd.concat(list_20w)
                    concat_20w_2 = concat_20w.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
                    list_total.append(concat_20w_2)
                    list_20w = []
                    m += 1
                    # print('==========', 'file:', k, '==========')
                    # print('==========', 'progress 1W:', j, 'progress 20w:', m, '==========')
            except Exception as e:
                error_list.append([k, e])

        if len(list_1w) != 0:
            concat_1w_rest = pd.concat(list_1w)
            concat_1W_rest_2 = concat_1w_rest.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
            list_total.append(concat_1W_rest_2)

        if len(list_20w) != 0:
            concat_20w_rest = pd.concat(list_20w)
            concat_20w_rest_2 = concat_20w_rest.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
            list_total.append(concat_20w_rest_2)

        tree_concat = pd.concat(list_total)
        tree_concat2 = tree_concat.drop_duplicates(subset=['time', 'places', 'next_places'], keep='first')
        # print(len(tree_concat), len(tree_concat2))
        tree_concat3 = tree_concat2.sort_values(by=['time', 'places', 'next_places'], ascending=True)

        # give each node a  unique index
        tree_concat3['tree_index'] = list(range(0, len(tree_concat3)))
        # tree_concat3.to_csv(done_dir + '//demo_tree_index.csv', index=False)
        error_df = pd.DataFrame(error_list)
        # error_df.to_csv('3_4_except_error.csv', index=False)
        # print(1)
        return tree_concat3, error_df

    def merge_tree(self, save_support_tree=True):

        data, _ = self.support_tree()

        data.replace('O_0', 'O', inplace=True)
        data.replace('O_1', 'O', inplace=True)
        data.replace('O_2', 'O', inplace=True)
        data.replace('O_3', 'O', inplace=True)
        data.replace('O_4', 'O', inplace=True)
        data.replace('O_5', 'O', inplace=True)
        data.replace('O_6', 'O', inplace=True)
        data.replace('O_7', 'O', inplace=True)
        data.replace('O_8', 'O', inplace=True)
        data.replace('O_9', 'O', inplace=True)
        data.replace('O_10', 'O', inplace=True)
        data.replace('O_11', 'O', inplace=True)
        data.replace('O_12', 'O', inplace=True)
        data.replace('O_13', 'O', inplace=True)
        data.replace('O_14', 'O', inplace=True)
        data.replace('O_15', 'O', inplace=True)
        data.replace('O_16', 'O', inplace=True)
        data.replace('O_17', 'O', inplace=True)
        data.replace('O_18', 'O', inplace=True)
        data.replace('O_19', 'O', inplace=True)
        data.replace('O_20', 'O', inplace=True)
        data.replace('O_21', 'O', inplace=True)
        data.replace('O_22', 'O', inplace=True)
        data.replace('O_23', 'O', inplace=True)
        data.replace('O_24', 'O', inplace=True)
        data.replace('O_25', 'O', inplace=True)
        data.replace('O_26', 'O', inplace=True)
        data.replace('O_27', 'O', inplace=True)
        data.replace('O_28', 'O', inplace=True)
        data.replace('O_29', 'O', inplace=True)
        data.replace('O_30', 'O', inplace=True)

        list_df = []
        for key, item in data.groupby(by='time'):
            time = key
            df_one = item
            df_one2 = df_one.drop_duplicates(subset=['places', 'next_places'], keep='first')
            list_df.append(df_one2)

        df_final = pd.concat(list_df, axis=0)

        df_final2 = df_final.sort_values(by=['time', 'places', 'next_places'], ascending=True)
        df_final2['tree_index'] = list(range(0, len(df_final2)))
        if save_support_tree:
            # print('Saving support tree...')
            df_final2.to_csv(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv', sep=',', encoding='utf-8')
        return df_final2

    def pattern_probability_matrix(self, generate_support_tree=False, raw_gps_file=None):

        self.raw_gps_file = raw_gps_file
        raw_gps_file_remark = self.raw_gps_file
        if generate_support_tree:
            df_total_tree_index = self.merge_tree()
        else:
            if os.path.exists(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv'):
                # print('Loading existing support tree...')
                df_total_tree_index = pd.read_csv(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv')
            else:
                print('No existing support tree...\nGenerating support tree...')
                df_total_tree_index = self.merge_tree()

        self.raw_gps_file = raw_gps_file_remark
        df = self.extract_life_pattern(self.raw_gps_file)

        df.replace('O_0', 'O', inplace=True)
        df.replace('O_1', 'O', inplace=True)
        df.replace('O_2', 'O', inplace=True)
        df.replace('O_3', 'O', inplace=True)
        df.replace('O_4', 'O', inplace=True)
        df.replace('O_5', 'O', inplace=True)
        df.replace('O_6', 'O', inplace=True)
        df.replace('O_7', 'O', inplace=True)
        df.replace('O_8', 'O', inplace=True)
        df.replace('O_9', 'O', inplace=True)
        df.replace('O_10', 'O', inplace=True)
        df.replace('O_11', 'O', inplace=True)
        df.replace('O_12', 'O', inplace=True)
        df.replace('O_13', 'O', inplace=True)
        df.replace('O_14', 'O', inplace=True)
        df.replace('O_15', 'O', inplace=True)
        df.replace('O_16', 'O', inplace=True)
        df.replace('O_17', 'O', inplace=True)
        df.replace('O_18', 'O', inplace=True)
        df.replace('O_19', 'O', inplace=True)
        df.replace('O_20', 'O', inplace=True)
        df.replace('O_21', 'O', inplace=True)
        df.replace('O_22', 'O', inplace=True)
        df.replace('O_23', 'O', inplace=True)
        df.replace('O_24', 'O', inplace=True)
        df.replace('O_25', 'O', inplace=True)
        df.replace('O_26', 'O', inplace=True)
        df.replace('O_27', 'O', inplace=True)
        df.replace('O_28', 'O', inplace=True)
        df.replace('O_29', 'O', inplace=True)
        df.replace('O_30', 'O', inplace=True)

        user_list = []
        for key_day, item_day in df.groupby(by='date'):
            df_1u_1day = item_day

            date = key_day
            df_one_user2 = df_1u_1day[['time', 'places', 'next_places']].copy()
            df_one_user2.loc[:, 'attribute'] = 1

            df_tree_total2 = df_total_tree_index[['tree_index', 'time', 'places', 'next_places']].copy()
            df_user_matrix = pd.merge(df_tree_total2, df_one_user2, how='left')

            df_user_matrix2 = df_user_matrix.fillna(0)
            df_user_matrix3 = df_user_matrix2[['tree_index', 'attribute']].copy()
            user_list.append(df_user_matrix3['attribute'])

        df_user_concat = pd.concat(user_list, axis=1)
        aaa = df_user_concat.mean(1)
        # print(self.user_id)

        return aaa

    def NMF_average(self, save_results=True, save_visualization=True, raw_gps_folder=None, raw_gps_file=None):

        self.raw_gps_file = raw_gps_file
        self.raw_gps_folder = raw_gps_folder

        if self.raw_gps_file is None:
            print('No individual gps file given...')
            if self.raw_gps_folder is None:
                print('Loading sample ZDC GPS folder...')
                self.raw_gps_folder = '../sample/'

        # filepath_list = []
            filename_list = glob.glob(self.raw_gps_folder + '*.csv')
        else:
            filename_list = [self.raw_gps_file]
        if self.initialize:
            filename_list = glob.glob(self.raw_gps_folder + '*.csv')

        user_matrix_list = []
        user_id_list = []
        for i in range(len(filename_list)):
            self.raw_gps_file = filename_list[i]
            # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
            # print(i, self.raw_gps_file)
            # data = pd.read_csv(filepath_list[i], index_col=[0])
            data = self.pattern_probability_matrix(raw_gps_file=self.raw_gps_file)
            user_matrix_list.append(np.array(data).flatten('F'))
            user_id_list.append(self.user_id)

        # non-negative matrix factorization
        user_matrix1 = np.array(user_matrix_list)
        # print(0)
        user_matrix = np.transpose(user_matrix1)
        # print(1)
        model = NMF(n_components=3, init='nndsvda')
        # print(2)

        w = model.fit_transform(user_matrix)

        H = model.components_

        df_w = pd.DataFrame(w)
        df_h = pd.DataFrame(H)

        # filename_df = pd.DataFrame(filename_list)
        filename_df = pd.DataFrame(user_id_list)
        if save_results:
            # print('Saving NMF results...\n')
            df_w.to_csv(self.NMF_results_folder + 'W_multiple_HW_single_O_total_day.csv', index=False)
            df_h.to_csv(self.NMF_results_folder + 'H_multiple_HW_single_O_total_day.csv', index=False)
            filename_df.to_csv(self.NMF_results_folder + 'filename_multiple_HW_single_O_total_day.csv', index=False)

        # visualization
        if save_visualization:
            df_H = df_h.T

            df_H['user_id'] = user_id_list
            df_H.columns = ['x', 'y', 'z', 'user_id']

            fig = plt.figure(figsize=(8, 5), dpi=300)
            ax = Axes3D(fig, auto_add_to_figure=False)
            fig.add_axes(ax)
            ax.scatter(df_H['x'].values, df_H['y'].values, df_H['z'].values, s=20, marker='o')
            ax.view_init(elev=35, azim=45)
            plt.savefig(self.NMF_results_folder + 'demo_result.png')
            plt.close()
            # print('Visualization saved...\n')
        return df_w, df_h, filename_df

    def mapping_in_space(self, new_gps_data, save_results=False):

        if os.path.exists(self.NMF_results_folder + 'W_multiple_HW_single_O_total_day.csv'):
            print('Loading existing meta file...')
            df_meta = pd.read_csv(self.NMF_results_folder + 'W_multiple_HW_single_O_total_day.csv')
        else:
            print('Generating meta file...')
            df_meta, df_h, filename = self.NMF_average()
        # df_meta = pd.read_csv(NMF_dir + '//2013_W_multiple_HW_single_O_total_day.csv')
        df_meta.columns = ['meta1', 'meta2', 'meta3']
        meta1 = np.array(df_meta['meta1'])
        meta2 = np.array(df_meta['meta2'])
        meta3 = np.array(df_meta['meta3'])

        x = [meta1, meta2, meta3]
        p0 = [0.01, 0.01, 0.01]

        if os.path.isfile(new_gps_data):
            self.raw_gps_file = new_gps_data
            # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
            df_individual = self.pattern_probability_matrix(raw_gps_file=self.raw_gps_file)
            y = df_individual.values.copy()
            para = least_squares(self.error, p0, args=(x, y), bounds=(0, np.inf))  # 进行拟合;args() 中是除了初始值之外error() 中的所有参数的集合输入。
            xyz = para.x
            # result_record_list = [self.id_, xyz[0], xyz[1], xyz[2]]
            # result = pd.DataFrame(result_record_list, columns=['2011_user_ID', 'x', 'y', 'z'])
            result = pd.DataFrame({'user_id': [self.user_id], 'x': [xyz[0]], 'y': [xyz[1]], 'z': [xyz[2]]})
            if save_results:
                # print('Saving results...\n')
                result.to_csv(self.NMF_results_folder + '2011_mapping_coor_has_constraint.csv', index=False)
            # print('finish')
        else:
            self.raw_gps_folder = new_gps_data
            runlist = glob.glob(self.raw_gps_folder + '*.csv')

            result_record_list = []
            for i in tqdm(range(len(runlist))):
                self.raw_gps_file = runlist[i]
                # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
                df_individual = self.pattern_probability_matrix(raw_gps_file=self.raw_gps_file)
                y = df_individual.values.copy()
                para = least_squares(self.error, p0, args=(x, y),
                                     bounds=(0, np.inf))  # 进行拟合;args() 中是除了初始值之外error() 中的所有参数的集合输入。
                xyz = para.x
                result_record_list.append([self.user_id, xyz[0], xyz[1], xyz[2]])
            result = pd.DataFrame(result_record_list, columns=['user_ID', 'x', 'y', 'z'])
            if save_results:
                # print('Saving results...\n')
                result.to_csv(self.NMF_results_folder + '2011_mapping_coor_has_constraint.csv', index=False)
            # print('finish')
        return result

    def Fun(self, p, meta):  # 定义拟合函数形式
        x, y, z = p
        meta1, meta2, meta3 = meta
        LP_simulate = x * meta1 + y * meta2 + z * meta3

        return LP_simulate

    def error(self, p, meta, LP_actual):  # 拟合残差;这里之所以没有平方是应为在拟合函数的内部进行，这里不显式的表示。
        return self.Fun(p, meta) - LP_actual

    def clustering(self, save_results=False, save_visualization=False, n_clusters=2, n_clusters_max=30, save_model=True, new_gps_data=None):

        checkpoint = all([os.path.exists(self.NMF_results_folder + 'W_multiple_HW_single_O_total_day.csv'),
                          os.path.exists(self.NMF_results_folder + 'H_multiple_HW_single_O_total_day.csv'),
                          os.path.exists(self.NMF_results_folder + 'filename_multiple_HW_single_O_total_day.csv')])
        if checkpoint:
            # print('Loading existing files...\n')
            df_w = pd.read_csv(self.NMF_results_folder + 'W_multiple_HW_single_O_total_day.csv')
            df_H0 = pd.read_csv(self.NMF_results_folder + 'H_multiple_HW_single_O_total_day.csv')
            filename_list = pd.read_csv(self.NMF_results_folder + 'filename_multiple_HW_single_O_total_day.csv')
        else:
            print('Generating clustering results...\n')
            df_w, df_H0, filename_list = self.NMF_average()
        filename_list.columns = ['user_id']
        filename0 = filename_list.reset_index(drop=True)

        df_H1 = df_H0.T.copy()
        df_H2 = df_H1.reset_index(drop=True)
        df_H2.columns = ['x', 'y', 'z']
        df_H2['user_id'] = filename_list['user_id'].copy()

        all_df_point = df_H2[["x", "y", "z"]]
        # print(1)
        X = all_df_point.to_numpy(copy=True)
        # print(2)

        # KMeans
        if self.initialize:
            kmeanModel = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(X)

            a = pd.Series(kmeanModel.labels_)
            b = kmeanModel.cluster_centers_
            data_a = pd.concat([df_H2, a], axis=1)
            data_a.columns = ['x', 'y', 'z', 'user_id', 'label']
            if save_results:
                data_a.to_csv(self.clustering_results_folder + 'demo_' + str(n_clusters) + '_KMeans.csv', index=False)
            # print(1)
            if save_model:
                joblib.dump(kmeanModel, self.clustering_results_folder + 'demo_kmeanModel.pkl')

            # visualization
            if save_visualization:
                cdict = {-1: 'black', 0: 'goldenrod', 1: "sandybrown", 2: 'yellowgreen', 3: "tomato", 4: 'plum', 5: 'steelblue',
                         6: 'lightskyblue',
                         7: 'lime', 8: "purple", 9: "indigo", 10: "peru", 11: 'sandybrown', 12: "fuchsia", 13: "red",
                         14: "limegreen", 15: 'red', 16: 'deeppink'}

                scatter_x = np.array(data_a['x'])
                scatter_y = np.array(data_a['y'])
                scatter_z = np.array(data_a['z'])
                group = np.array(data_a['label'])

                fig = plt.figure(figsize=(8, 5), dpi=200)
                ax = Axes3D(fig, auto_add_to_figure=False)
                fig.add_axes(ax)
                for g in np.unique(group):
                    ix = np.where(group == g)
                    ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=cdict[g], label=g, s=20)
                ax.legend()
                ax.view_init(elev=35, azim=45)

                plt.savefig(self.clustering_results_folder + 'demo_7KMEANS.png')
                # plt.show()
                plt.close()
        # print(1)
        else:
            if os.path.exists(self.clustering_results_folder + 'demo_kmeanModel.pkl'):
                kmeanModel = joblib.load(self.clustering_results_folder + 'demo_kmeanModel.pkl')
            else:
                raise 'No pre-trained Kmeans model exists...'
            df_2011 = self.mapping_in_space(new_gps_data=new_gps_data)

            all_df_point_2011 = df_2011[["x", "y", "z"]]
            X_2013 = all_df_point_2011.to_numpy(copy=True)

            label_2011 = kmeanModel.predict(X_2013)

            data_a = pd.concat([df_2011, pd.Series(label_2011)], axis=1)

            data_a.columns = ['user_id', 'x', 'y', 'z', 'LP_label']
            if save_results:
                data_a.to_csv(self.clustering_results_folder + '2011_7_cluster_KMeans.csv', index=False)
            if save_visualization:
                cdict = {-1: 'black', 0: 'goldenrod', 1: "sandybrown", 2: 'yellowgreen', 3: "tomato", 4: 'plum', 5: 'steelblue',
                         6: 'lightskyblue',
                         7: 'lime', 8: "purple", 9: "indigo", 10: "peru", 11: 'sandybrown', 12: "fuchsia", 13: "red",
                         14: "limegreen", 15: 'red', 16: 'deeppink'}

                # data_a_2011 = pd.read_csv('9_conduct_same_clustor_as2013//2011_7_cluster_KMeans.csv')

                scatter_x = np.array(data_a['x'])
                scatter_y = np.array(data_a['y'])
                scatter_z = np.array(data_a['z'])
                group = np.array(data_a['LP_label'])

                fig = plt.figure(figsize=(8, 5), dpi=200)
                ax = Axes3D(fig,auto_add_to_figure=False)
                fig.add_axes(ax)
                for g in np.unique(group):
                    ix = np.where(group == g)
                    ax.scatter(scatter_x[ix], scatter_y[ix], scatter_z[ix], c=cdict[g], label=g, s=10)
                ax.legend()
                # ax.set_xlim((0, 0.2))
                # ax.set_ylim((0, 0.2))
                # ax.set_zlim((0, 0.2))
                ax.view_init(elev=35, azim=45)
                #
                plt.savefig(self.clustering_results_folder + '2011_KMEANS_as2013.png')
                # plt.show()
                plt.close()
                # print(1)
        return data_a

    def plus_home_work_location(self, save_results=False, new_gps_data=None):

        if self.initialize:
            df_clustering_result = self.clustering()
        else:
            if os.path.exists(self.clustering_results_folder + 'demo_2_KMeans.csv'):
                df_clustering_result = pd.read_csv(self.clustering_results_folder + 'demo_2_KMeans.csv')
            else:
                df_clustering_result = self.clustering(new_gps_data=new_gps_data)

        for i in range(len(df_clustering_result)):

            # print('processing:', i)

            user_ID = str(df_clustering_result.loc[i, 'user_id']).zfill(8)

            # if os.path.exists('./')

            # df_home_work = pd.read_csv(home_work_location_dir + '//' + user_ID + '_great_tokyo_labeled.csv')
            self.raw_gps_file = self.raw_gps_folder + user_ID + '.csv'
            # self.raw_gps_file = user_ID
            # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
            df_home_work = self.detect_home_work(raw_gps_file=self.raw_gps_file)

            df_home = df_home_work[df_home_work['home_label_order'] == 0]

            df_work = df_home_work[df_home_work['work_label_order'] == 0]

            if len(df_home) != 0:
                home_lng = df_home['home_lon'].mean()
                home_lat = df_home['home_lat'].mean()

            if len(df_home) == 0:
                home_lng = -1
                home_lat = -1

            if len(df_work) != 0:
                work_lng = df_work['work_lon'].mean()
                work_lat = df_work['work_lat'].mean()

            if len(df_work) == 0:
                work_lng = -1
                work_lat = -1

            df_clustering_result.loc[i, 'home_lat'] = home_lat
            df_clustering_result.loc[i, 'home_lng'] = home_lng
            df_clustering_result.loc[i, 'work_lat'] = work_lat
            df_clustering_result.loc[i, 'work_lng'] = work_lng

        if save_results:
            if self.initialize:
                df_clustering_result.to_csv(self.clustering_results_folder + '8_ID_XYZ_groupID_homeworklocation.csv')
            else:
                df_clustering_result.to_csv(self.clustering_results_folder + 'New_ID_XYZ_groupID_homeworklocation.csv')

        # print('finish')

    def csv2Point(self, data, LON, LAT, coordinate):
        '''
        FILE_PATH: the path of csv file that records the longitude and latitude of home/work place.

        ID: the column name of user ID.

        LON: the column name of longitude.

        LAT: the column name of latitude.
        '''

        data_point = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data[LON], data[LAT]), crs=coordinate)

        return data_point

    def Point2Name(self, SHP_PATH, POINT, COLUMNS, coordinate):
        '''
        SHP_PATH: the path of administrative boundary shapefile.

        POINT: geodataframe of points.

        COLUMNS: the columns you wish to keep.
        '''

        chome = gpd.read_file(SHP_PATH).to_crs({'init': coordinate})

        COLUMNS.append('geometry')
        chome = chome[COLUMNS]
        join_point = gpd.sjoin(POINT, chome, how='left', op='intersects', lsuffix='left', rsuffix='right')
        join_point = join_point.drop(['index_right', 'geometry'], axis=1)
        return join_point

    def generate_key_point_table(self, save_results=False, new_gps_data=None):

        if os.path.exists(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv'):
            df_totaltree = pd.read_csv(self.support_tree_folder + 'demo_tree_index_multiple_HW_single_O.csv')
        else:
            df_totaltree = self.merge_tree()
        # df_totaltree = pd.read_csv(tree_index_dir + '//the_great_tokyo_total_tree_index.csv')
        significant_place_list0 = df_totaltree['places'].tolist() + df_totaltree['next_places'].tolist()
        significant_place_list = list(set(significant_place_list0))
        significant_place_df = pd.DataFrame(significant_place_list, columns=['place_name'])
        a = significant_place_df['place_name'].str.split("_", expand=True)
        a.columns = ['place', 'order']
        a.loc[:, 'order'] = a['order'].fillna(0)
        a['order'] = a['order'].astype('int')
        a = a.sort_values(['place', 'order'], ascending=[True, True])
        b = pd.concat([a[a['place'] == 'H'], a[a['place'] == 'W'], a[a['place'] == 'O']])
        b = b.reset_index(drop=True)

        dic_ = {"H": 'home', "W": 'work', 'O': "other"}

        # 2.find key point
        chome = gpd.read_file('./gis/h12ka13.shp')
        chome = chome[['KEY_CODE', 'PREF_NAME', 'CITY_NAME', 'S_NAME', 'geometry']]
        if self.initialize:
            if os.path.exists(self.clustering_results_folder + 'demo_2_KMeans.csv'):
                df_clustering_result = pd.read_csv(self.clustering_results_folder + 'demo_2_KMeans.csv')
            else:
                df_clustering_result = self.clustering()
        else:
            df_clustering_result = self.clustering(new_gps_data=new_gps_data)

        for i in range(len(df_clustering_result)):

            # print('processing:', i)

            user_ID = str(df_clustering_result.loc[i, 'user_id']).zfill(8)

            # df_home_work = pd.read_csv(home_work_location_dir + '//' + user_ID + '_great_tokyo_labeled.csv')
            # self.raw_gps_file = user_ID
            self.raw_gps_file = self.raw_gps_folder + user_ID + '.csv'
            # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
            df_home_work = self.detect_home_work(raw_gps_file=self.raw_gps_file)

            individual_key_point_df = b.copy()
            individual_key_point_df['user_ID'] = user_ID

            for k in range(len(individual_key_point_df)):
                place = individual_key_point_df.loc[k, 'place']
                order = individual_key_point_df.loc[k, 'order']

                individual_key_point_df.loc[k, 'significant_place'] = place + '_' + str(order)
                temp = df_home_work[df_home_work[dic_[place] + '_label_order'] == order]

                if len(temp) != 0:
                    Lng = temp[dic_[place] + '_lon'].mean()
                    Lat = temp[dic_[place] + '_lat'].mean()
                    mesh_code = ju.to_meshcode(Lat, Lng, 5)

                    individual_key_point_df.loc[k, 'Lng'] = Lng
                    individual_key_point_df.loc[k, 'Lat'] = Lat
                    individual_key_point_df.loc[k, 'mesh_code'] = mesh_code

            individual_key_point_df = individual_key_point_df.fillna(-1)

            point = self.csv2Point(individual_key_point_df, 'Lng', 'Lat', 'EPSG:4326')
            join_point = self.Point2Name('./gis/h12ka13.shp', point, ['KEY_CODE', 'PREF_NAME', 'CITY_NAME', 'S_NAME'], 'EPSG:4326')
            join_point = join_point.fillna(-1)
            if save_results:
                if not os.path.exists('./key_point_table/'):
                    os.mkdir('./key_point_table/')
                join_point.to_csv('./key_point_table/' + str(self.user_id) + '_key_point_table.csv', index=False)
        return join_point

    def generate_group_HWO_join_area2(self, save_results=False, new_gps_data=None):
        if self.initialize:
            if os.path.exists(self.clustering_results_folder + 'demo_2_KMeans.csv'):
                df = pd.read_csv(self.clustering_results_folder + 'demo_2_KMeans.csv')
            else:
                df = self.clustering()
        else:
            df = self.clustering(new_gps_data=new_gps_data)
        # 或者df = pd.read_csv('9_conduct_same_cluster_as2013//2011_7_cluster_KMeans.csv')

        for i in range(len(df)):
            user_ID = str(df.loc[i, 'user_id']).zfill(8)
            # df_1_user_stay = pd.read_csv(
            #     '2_great_tokyo_detect_home_work_order' + '//' + userID + '_great_tokyo_labeled.csv')
            # self.raw_gps_file = user_ID
            self.raw_gps_file = self.raw_gps_folder + user_ID + '.csv'
            # self.id_ = self.raw_gps_file.split('/')[-1].split('.')[0]
            df_1_user_stay = self.detect_home_work(self.raw_gps_file)
            df.loc[i, 'home_Lat_WGS84'] = df_1_user_stay[df_1_user_stay['home_label_order'] == 0]['home_lat'].mean()
            df.loc[i, 'home_Lng_WGS84'] = df_1_user_stay[df_1_user_stay['home_label_order'] == 0]['home_lon'].mean()
            df.loc[i, 'work_Lat_WGS84'] = df_1_user_stay[df_1_user_stay['work_label_order'] == 0]['work_lat'].mean()
            df.loc[i, 'work_Lng_WGS84'] = df_1_user_stay[df_1_user_stay['work_label_order'] == 0]['work_lon'].mean()

        df['home_Lat_WGS84'].fillna(-1, inplace=True)
        df['home_Lng_WGS84'].fillna(-1, inplace=True)
        df_point = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['home_Lng_WGS84'], df['home_Lat_WGS84']),
                                    crs='EPSG:4326')
        Tokyo_map = gpd.read_file('./gis/tokyo_area_satatistic.shp').to_crs({'init': 'EPSG:4326'})
        Tokyo_map = Tokyo_map[['N03_001', 'N03_003', 'N03_004', 'geometry']]
        join_point = gpd.sjoin(df_point, Tokyo_map, how='left', op='intersects', lsuffix='left', rsuffix='right')
        join_point = join_point.drop(['index_right', 'geometry'], axis=1)
        join_point = join_point.fillna(-1)
        if save_results:
            join_point.to_csv('./11_demo_7_group_HWO_join_area2.csv', index=False)

