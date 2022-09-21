import datetime
import multiprocessing
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

R = 6371008

class brake:
    def __init__(self,
                 file_dir_1 = "D:/paper/brake/data/interpolation",
                 figure_save_path_1 = "D:/paper/brake/figure/velocity/",
                 file_dir_23 = "D:/paper/brake/data/all_trajectory.csv",
                 figure_save_path_3 = "D:/paper/brake/figure/",
                 initilize = False
                 ):

        self.file_dir_1 = file_dir_1
        self.figure_save_path_1 = figure_save_path_1
        self.file_dir_23 = file_dir_23
        self.figure_save_path_3 = figure_save_path_3
        self.initilize = initilize

        if self.initilize:
            self.main_1()
            self.main_2()
            self.main_3()


    def main_funcition_1(self, trajectory_file_path, j):
        data = pd.read_csv(trajectory_file_path[j], index_col=[0])
        if len(data) > 5:
            data["time"] = pd.to_datetime(data["time"])
            data["lon"] = pd.to_numeric(data["lon"])
            data["lat"] = pd.to_numeric(data["lat"])
            data["time_interval"] = pd.to_numeric(data["time_interval"])
            data["vehicle_type"] = pd.to_numeric(data["vehicle_type"])
            key = data.loc[0, "routine_ID"]
            item = data.copy()

            item.loc[0, "distance"] = 0
            item["distance"] = pd.to_numeric(item["distance"])
            raw_velocity_list = item.loc[1:, "distance"]
            raw_velocity_list = raw_velocity_list.reset_index(drop=True)
            raw_acceleration_list = [raw_velocity_list[i + 1] - raw_velocity_list[i] for i in
                                     range(len(raw_velocity_list) - 1)]
            raw_acceleration_list = np.array(raw_acceleration_list)
            velocity_list_1 = np.zeros(len(raw_velocity_list))
            velocity_list_1[0] = raw_velocity_list[0]
            for i in range(len(raw_acceleration_list)):
                if abs(raw_acceleration_list[i]) <= 6:
                    velocity_list_1[i + 1] = raw_velocity_list[i + 1]
                else:
                    if i == (len(raw_acceleration_list) - 1):
                        velocity_list_1[i + 1] = velocity_list_1[i] + raw_acceleration_list[i - 1]
                    else:
                        try:
                            velocity_list_1[i + 1] = (velocity_list_1[i] + velocity_list_1[i + 2]) / 2  # 差值补空
                        except:
                            sys.exit("got error in filtering!")
            if len(velocity_list_1) > 10000:
                length_for_show = int(len(velocity_list_1) * 0.01)
                x = np.arange(length_for_show)
                raw_velocity_list_for_show = raw_velocity_list[:length_for_show]
                velocity_list_1_for_show = velocity_list_1[:length_for_show]
            elif len(velocity_list_1) < 10000 and len(velocity_list_1) > 1000:
                length_for_show = int(len(velocity_list_1) * 0.1)
                x = np.arange(length_for_show)
                raw_velocity_list_for_show = raw_velocity_list[:length_for_show]
                velocity_list_1_for_show = velocity_list_1[:length_for_show]
            else:
                length_for_show = int(len(velocity_list_1))
                x = np.arange(length_for_show)
                raw_velocity_list_for_show = raw_velocity_list[:int(length_for_show + 1)]
                velocity_list_1_for_show = velocity_list_1[:int(length_for_show + 1)]

            state_variable = np.asarray([velocity_list_1])
            state_variable = np.transpose(state_variable)
            # kalman
            kf = KalmanFilter(n_dim_state=1,
                              n_dim_obs=1,
                              em_vars=["transition_offsets",
                                       'observation_offsets',
                                       'transition_covariance',
                                       'observation_covariance',
                                       'initial_state_mean',
                                       'initial_state_covariance'])
            try:
                kf = kf.em(state_variable, em_vars=["transition_offsets",
                                                    'observation_offsets',
                                                    'transition_covariance',
                                                    'observation_covariance',
                                                    'initial_state_mean',
                                                    'initial_state_covariance'])
            except:
                print(1)
            (smoothed_state_means, smoothed_state_covariances) = kf.smooth(state_variable)
            # (filtered_state_means, filtered_state_covariances) = kf.filter(state_variable)

            if len(velocity_list_1) > 10000:
                length_for_show = int(len(velocity_list_1) * 0.01)
                smoothed_original_velocity = smoothed_state_means[:, 0]
                smoothed_original_velocity_for_show = smoothed_original_velocity[:length_for_show]
            elif len(velocity_list_1) < 10000 and len(velocity_list_1) > 1000:
                length_for_show = int(len(velocity_list_1) * 0.1)
                smoothed_original_velocity = smoothed_state_means[:, 0]
                smoothed_original_velocity_for_show = smoothed_original_velocity[:length_for_show]
            else:
                length_for_show = int(len(velocity_list_1))
                smoothed_original_velocity = smoothed_state_means[:, 0]
                smoothed_original_velocity_for_show = smoothed_original_velocity[:int(length_for_show + 1)]

            plt.figure()
            plt.plot(x, raw_velocity_list_for_show, 'g',
                     x, velocity_list_1_for_show, 'b',
                     x, smoothed_original_velocity_for_show, 'r', linewidth=0.8)
            plt.legend(["Raw velocity",
                        "no outlier velocity",
                        "smoothed velocity"])
            plt.xlabel('time (s)')
            plt.ylabel('velocity(m/s)')
            plt.title("velocity contrast_" + key)
            plt.savefig(self.figure_save_path_1 + "velocity contrast_" + key + ".jpg")

            plt.close('all')

        print("complete: {:.4%}".format(j / len(trajectory_file_path)))
        return 0

    def main_1(self):
        trajectory_file_path = []
        for root, dirs, files in os.walk(self.file_dir_1):
            for filename in files:
                trajectory_file_path.append(os.path.join(root, filename))
        trajectory_file_path_list = []
        for i in range(len(trajectory_file_path)):
            trajectory_file_path_list.append((trajectory_file_path, i))
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(10) as pool:
            pool.starmap(self.main_funcition_1, trajectory_file_path_list)
        return

    def interpo(self, item, key):
        interpolated_item = pd.DataFrame(columns=['routine_ID', 'time', 'lon', 'lat', 'time_interval', 'distance',
                                                  'vehicle_type', "interpolation"])
        interpolated_item.loc[0, :] = item.loc[0, :]
        interpolated_item.loc[0, "interpolation"] = 0
        interpolated_item_i = 1
        for i in range(1, len(item)):
            if item.loc[i, "time_interval"] == 1:
                interpolated_item.loc[interpolated_item_i, :] = item.loc[i, :]
                interpolated_item.loc[interpolated_item_i, "interpolation"] = 0
                interpolated_item_i += 1
            else:
                interpolation_time = item.loc[i, "time_interval"] - 1
                interpolation_array = np.zeros([int(interpolation_time), 2])
                for j in range(len(interpolation_array)):
                    interpolation_array[j, 0] = item.loc[i - 1, "lon"] + (item.loc[i, "lon"] - item.loc[i - 1, "lon"]) / \
                                                item.loc[i, "time_interval"]
                    interpolation_array[j, 1] = item.loc[i - 1, "lat"] + (item.loc[i, "lat"] - item.loc[i - 1, "lat"]) / \
                                                item.loc[i, "time_interval"]
                # x = np.concatenate(([item.loc[i - 1, "lon"]], interpolation_array[:, 0], [item.loc[i, "lon"]]))
                # y = np.concatenate(([item.loc[i - 1, "lat"]], interpolation_array[:, 1], [item.loc[i, "lat"]]))
                for j in range(len(interpolation_array)):
                    interpolated_item.loc[interpolated_item_i, "routine_ID"] = key
                    interpolated_item.loc[interpolated_item_i, "time"] = interpolated_item.loc[
                                                                             interpolated_item_i - 1, "time"] + datetime.timedelta(
                        seconds=1)
                    interpolated_item.loc[interpolated_item_i, "lon"] = interpolation_array[j, 0]
                    interpolated_item.loc[interpolated_item_i, "lat"] = interpolation_array[j, 1]
                    interpolated_item.loc[interpolated_item_i, "time_interval"] = 1
                    interpolated_item.loc[interpolated_item_i, "distance"] = 0
                    interpolated_item.loc[interpolated_item_i, "vehicle_type"] = item.loc[0, "vehicle_type"]
                    interpolated_item.loc[interpolated_item_i, "interpolation"] = 1
                    interpolated_item_i += 1
        return interpolated_item

    def duankai_interpo(self, item, dijige, key):
        item = item.reset_index(drop=True)
        interpolated_item = pd.DataFrame(columns=['routine_ID', 'time', 'lon', 'lat', 'time_interval', 'distance',
                                                  'vehicle_type', "interpolation"])
        interpolated_item.loc[0, :] = item.loc[0, :]
        interpolated_item.loc[0, "routine_ID"] = key + "_" + str(dijige)
        interpolated_item.loc[0, "time_interval"] = 0
        interpolated_item.loc[0, "interpolation"] = 0
        interpolated_item_i = 1
        for i in range(1, len(item)):
            if item.loc[i, "time_interval"] == 1:
                interpolated_item.loc[interpolated_item_i, :] = item.loc[i, :]
                interpolated_item.loc[interpolated_item_i, "routine_ID"] = key + "_" + str(dijige)
                interpolated_item.loc[interpolated_item_i, "interpolation"] = 0
                interpolated_item_i += 1
            else:
                interpolation_time = item.loc[i, "time_interval"] - 1
                if interpolation_time < 0:
                    print(1)
                interpolation_array = np.zeros([int(interpolation_time), 2])
                for j in range(len(interpolation_array)):
                    interpolation_array[j, 0] = item.loc[i - 1, "lon"] + (item.loc[i, "lon"] - item.loc[i - 1, "lon"]) / \
                                                item.loc[i, "time_interval"]
                    interpolation_array[j, 1] = item.loc[i - 1, "lat"] + (item.loc[i, "lat"] - item.loc[i - 1, "lat"]) / \
                                                item.loc[i, "time_interval"]
                for j in range(len(interpolation_array)):
                    interpolated_item.loc[interpolated_item_i, "routine_ID"] = key + "_" + str(dijige)
                    interpolated_item.loc[interpolated_item_i, "time"] = interpolated_item.loc[
                                                                             interpolated_item_i - 1, "time"] + datetime.timedelta(
                        seconds=1)
                    interpolated_item.loc[interpolated_item_i, "lon"] = interpolation_array[j, 0]
                    interpolated_item.loc[interpolated_item_i, "lat"] = interpolation_array[j, 1]
                    interpolated_item.loc[interpolated_item_i, "time_interval"] = 1
                    interpolated_item.loc[interpolated_item_i, "distance"] = 0
                    interpolated_item.loc[interpolated_item_i, "vehicle_type"] = item.loc[0, "vehicle_type"]
                    interpolated_item.loc[interpolated_item_i, "interpolation"] = 1
                    interpolated_item_i += 1
        return interpolated_item

    def main_2(self):
        all_trajectory = pd.read_csv(self.file_dir_23, index_col=[0], low_memory=False)
        all_trajectory["lon"] = pd.to_numeric(all_trajectory["lon"])
        all_trajectory["lat"] = pd.to_numeric(all_trajectory["lat"])
        all_trajectory["time"] = pd.to_datetime(all_trajectory["time"])
        grouped_all_trajectory = all_trajectory.groupby(["routine_ID"])
        print(len(grouped_all_trajectory))
        jishu = 0
        for key, item in grouped_all_trajectory:
            item = item.sort_values(by="time")
            item = item.reset_index(drop=True)
            item.loc[0, "time_interval"] = 0
            for i in range(1, len(item)):
                item.loc[i, "time_interval"] = (item.loc[i, "time"] - item.loc[i - 1, "time"]).total_seconds()
            # if key == "00043fc14e59e334fed609e94d8ee1bf97ced4a7":
            #     print(1)
            duandian = item[item["time_interval"] > 10]
            if len(duandian) == 0:
                interpolated_item = self.interpo(item, key)
                # interpolation_x = np.array([item.loc[i - 1, "lon"], item.loc[i - 1, "lat"]])
                # interpolation_y = np.array([item.loc[i - 1, "lon"], item.loc[i - 1, "lat"]])
                interpolated_item.to_csv(
                    "D:/paper/brake/data/interpolation/interpolated_trajectory_" + interpolated_item.loc[
                        0, "routine_ID"] + ".csv")
            else:
                for i in range(len(duandian)):
                    if i == 0:
                        duankai_item = item.loc[:(duandian.index[i] - 1), :]
                    else:
                        duankai_item = item.loc[duandian.index[i - 1]:(duandian.index[i] - 1), :]
                    if len(duankai_item) > 1:
                        interpolated_item = self.duankai_interpo(duankai_item, i, key)
                        interpolated_item.to_csv(
                            "D:/paper/brake/data/interpolation/interpolated_trajectory_" + interpolated_item.loc[
                                0, "routine_ID"] + ".csv")
                duankai_item = item.loc[duandian.index[-1]:, :]
                if len(duankai_item) > 1:
                    interpolated_item = self.duankai_interpo(duankai_item, (i + 1), key)
                    interpolated_item.to_csv(
                        "D:/paper/brake/data/interpolation/interpolated_trajectory_" + interpolated_item.loc[
                            0, "routine_ID"] + ".csv")
            jishu += 1
            print("{:.4%}".format(jishu / len(grouped_all_trajectory)))
        return

    def calculate_dis(self, O_lat, O_lon, D_lat, D_lon):
        a = np.radians(np.asarray(O_lon, dtype=np.float64) - np.asarray(D_lon, dtype=np.float64))
        b = np.radians(np.asarray(O_lat, dtype=np.float64) - np.asarray(D_lat, dtype=np.float64))
        R = 6371008
        dis = 2 * R * np.arcsin(np.sqrt(np.sin(b/2) ** 2 + np.cos(np.radians(np.asarray(O_lat, dtype=np.float64))) * np.cos(np.radians(np.asarray(D_lat, dtype=np.float64))) * np.sin(a/2) ** 2))
        return dis

    def cal_new_coordinate(self, position1_x, position1_y, position2_x, position2_y, original_distance, velocity):
        if velocity == 0:
            return (position1_x, position1_y)
        if (position2_x >= position1_x) and (position2_y >= position1_y):
            return (position1_x + abs(position2_x - position1_x) / original_distance * velocity,
                    position1_y + abs(position2_y - position1_y) / original_distance * velocity)
        elif (position2_x >= position1_x) and (position2_y <= position1_y):
            return (position1_x + abs(position2_x - position1_x) / original_distance * velocity,
                    position1_y - abs(position2_y - position1_y) / original_distance * velocity)
        elif (position2_x <= position1_x) and (position2_y >= position1_y):
            return (position1_x - abs(position2_x - position1_x) / original_distance * velocity,
                    position1_y + abs(position2_y - position1_y) / original_distance * velocity)
        elif (position2_x <= position1_x) and (position2_y <= position1_y):
            return (position1_x - abs(position2_x - position1_x) / original_distance * velocity,
                    position1_y - abs(position2_y - position1_y) / original_distance * velocity)
        return

    def main_3(self):
        data = pd.read_csv(self.file_dir_23, low_memory=False)
        data.columns = ["routine_ID", "time", "lon", "lat", "time_diff", "distance_diff", "vehicle_type"]
        grouped_data = data.groupby(by="routine_ID")
        driver_number = 0
        for key, item in grouped_data:
            item = item.reset_index(drop=True)
            coordinate_df = pd.DataFrame(columns=["from_lon", "from_lat", "to_lon", "to_lat"])
            for i in range(len(item) - 1):
                coordinate_df.loc[i, "from_lon"] = item.loc[i, "lon"]
                coordinate_df.loc[i, "from_lat"] = item.loc[i, "lat"]
                coordinate_df.loc[i, "to_lon"] = item.loc[i + 1, "lon"]
                coordinate_df.loc[i, "to_lat"] = item.loc[i + 1, "lat"]
            raw_velocity_list = self.calculate_dis(coordinate_df["from_lat"], coordinate_df["from_lon"],
                                              coordinate_df["to_lat"],
                                              coordinate_df["to_lon"])

            distance_matrix = np.zeros([len(item)])
            distance_matrix[0] = 0
            for i in range(len(raw_velocity_list)):
                distance_matrix[i + 1] = np.sum(raw_velocity_list[:(i + 1)])

            raw_acceleration_list = [raw_velocity_list[i + 1] - raw_velocity_list[i] for i in
                                     range(len(raw_velocity_list) - 1)]
            raw_acceleration_list = np.array(raw_acceleration_list)
            velocity_list_1 = np.zeros(len(raw_velocity_list))
            velocity_list_1[0] = raw_velocity_list[0]
            # 去噪
            for i in range(len(raw_acceleration_list)):
                if abs(raw_acceleration_list[i]) <= 6:
                    velocity_list_1[i + 1] = raw_velocity_list[i + 1]
                else:
                    if i == (len(raw_acceleration_list) - 2):
                        velocity_list_1[i + 1] = velocity_list_1[i] + raw_acceleration_list[i - 1]
                    else:
                        velocity_list_1[i + 1] = (velocity_list_1[i] + velocity_list_1[i + 2]) / 2  # 差值补空

            x = np.arange(0, len(velocity_list_1))

            plt.figure()
            plt.plot(x, raw_velocity_list)
            plt.xlabel('time (s)')
            plt.ylabel('velocity(m/s)')
            plt.title("Raw velocity_" + str(driver_number))
            plt.savefig(self.figure_save_path_3 + "Raw velocity_" + str(driver_number) + ".jpg")

            plt.figure()
            plt.plot(x, velocity_list_1)
            plt.xlabel('time (s)')
            plt.ylabel('velocity(m/s)')
            plt.title("no outlier velocity_" + str(driver_number))
            plt.savefig(self.figure_save_path3 + "No outlier velocity_" + str(driver_number) + ".jpg")

            state_variable = np.asarray([velocity_list_1])
            state_variable = np.transpose(state_variable)
            # kalman
            kf = KalmanFilter(n_dim_state=1,
                              n_dim_obs=1,
                              em_vars=["transition_offsets",
                                       'observation_offsets',
                                       'transition_covariance',
                                       'observation_covariance',
                                       'initial_state_mean',
                                       'initial_state_covariance'])
            kf = kf.em(state_variable, em_vars=["transition_offsets",
                                                'observation_offsets',
                                                'transition_covariance',
                                                'observation_covariance',
                                                'initial_state_mean',
                                                'initial_state_covariance'])
            (smoothed_state_means, smoothed_state_covariances) = kf.smooth(state_variable)

            x = np.arange(0, len(smoothed_state_means))
            smoothed_original_velocity = smoothed_state_means[:, 0]

            plt.figure()
            plt.plot(x, smoothed_original_velocity, 'r')
            plt.title("smoothed velocity_" + str(driver_number))
            plt.savefig(self.figure_save_path_3 + "smoothed velocity_" + str(driver_number) + ".jpg")

            # smoothed_acceleration = np.zeros(len(smoothed_original_velocity))

            new_coordinate = pd.DataFrame(columns=["lon", "lat"])
            new_coordinate.loc[0, "lon"] = item.loc[0, "lon"]
            new_coordinate.loc[0, "lat"] = item.loc[0, "lat"]
            for i in range(len(smoothed_original_velocity)):
                a = np.sum(smoothed_original_velocity[:(i + 1)])
                if a < 0:
                    new_coordinate.loc[i + 1, "lon"] = new_coordinate.loc[i, "lon"]
                    new_coordinate.loc[i + 1, "lat"] = new_coordinate.loc[i, "lat"]
                    continue
                array_position1 = np.where(distance_matrix <= a)[0][-1]
                if array_position1 == (len(distance_matrix) - 1):
                    array_position2 = array_position1
                    array_position1 = array_position1 - 1
                    remained_distance = a - distance_matrix[array_position1]
                    original_distance = distance_matrix[array_position2] - distance_matrix[array_position1]
                    position = self.cal_new_coordinate(item.loc[array_position1, "lon"], item.loc[array_position1, "lat"],
                                                  item.loc[array_position2, "lon"], item.loc[array_position2, "lat"],
                                                  original_distance, remained_distance)
                    new_coordinate.loc[i + 1, "lon"] = position[0]
                    new_coordinate.loc[i + 1, "lat"] = position[1]
                else:
                    array_position2 = np.where(distance_matrix >= a)[0][0]
                    remained_distance = a - distance_matrix[array_position1]
                    original_distance = distance_matrix[array_position2] - distance_matrix[array_position1]
                    position = self.cal_new_coordinate(item.loc[array_position1, "lon"], item.loc[array_position1, "lat"],
                                                  item.loc[array_position2, "lon"], item.loc[array_position2, "lat"],
                                                  original_distance, remained_distance)
                    new_coordinate.loc[i + 1, "lon"] = position[0]
                    new_coordinate.loc[i + 1, "lat"] = position[1]
            new_coordinate.to_csv("D:/paper/brake/filtered" + str(driver_number) + ".csv")
            print("{:.4%}".format(driver_number / len(grouped_data)))
            driver_number += 1
        return

if __name__ == '__main__':
    brake(initialize = True)
