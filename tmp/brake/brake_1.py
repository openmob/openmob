import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import sys
import os
import multiprocessing
def calculate_dis(O_lat, O_lon, D_lat, D_lon):
    a = np.radians(np.asarray(O_lon, dtype=np.float64) - np.asarray(D_lon, dtype=np.float64))
    b = np.radians(np.asarray(O_lat, dtype=np.float64) - np.asarray(D_lat, dtype=np.float64))
    dis = 2 * R * np.arcsin(np.sqrt(np.sin(b/2) ** 2 + np.cos(np.radians(np.asarray(O_lat, dtype=np.float64))) * np.cos(np.radians(np.asarray(D_lat, dtype=np.float64))) * np.sin(a/2) ** 2))
    return dis


file_dir = "D:/paper/brake/data/interpolation"
# raw_velocity_figure_save_path = "D:/paper/brake/figure/raw velocity/"
# no_outlier_velocity_figure_save_path = "D:/paper/brake/figure/no outlier velocity/"
figure_save_path = "D:/paper/brake/figure/velocity/"
LDV_highway = 0.1
LDV_urban_street_canyon = 0.2
HDV_highway = 0.3
HDV_urban_street_canyon = 0.4
R = 6371008
brake_deceleration_threshold = 1
number_of_parrel = 10

def main_function(trajectory_file_path, j):
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
        raw_acceleration_list = [raw_velocity_list[i + 1] - raw_velocity_list[i] for i in range(len(raw_velocity_list) - 1)]
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
                        velocity_list_1[i + 1] = (velocity_list_1[i] + velocity_list_1[i + 2])/2 #差值补空
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

        # smoothed_wavelet_velocity = smoothed_state_means[:, 1]
        # filtered_velocity = np.sqrt((filtered_state_means[:, 0] ** 2 + filtered_state_means[:, 1] ** 2))

        # plt.figure(1)
        # plt.plot(x, filtered_velocity, 'bo')
        # plt.show()
        # x, velocity_list_2, 'b--',
        # plt.figure()
        # plt.plot()
        # plt.xlabel('time (s)')
        # plt.ylabel('velocity(m/s)')
        # plt.title("_" + key)
        # plt.savefig(raw_velocity_figure_save_path + "raw velocity_" + key +".jpg")
        # plt.close()
        #
        # plt.figure()
        # plt.plot()
        # plt.title("_" + key)
        # plt.savefig(no_outlier_velocity_figure_save_path + "no outlier velocity_" + key + ".jpg")

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
        plt.savefig(figure_save_path + "velocity contrast_" + key + ".jpg")

        plt.close('all')

        # new_coordinate = pd.DataFrame(columns=["lon", "lat"])
        # new_coordinate.loc[0, "lon"] = item.loc[0, "lon"]
        # new_coordinate.loc[0, "lat"] = item.loc[0, "lat"]
        # for i in range(len(smoothed_original_velocity)):
        #     a = np.sum(smoothed_original_velocity[:(i + 1)])
        #     if a < 0:
        #         new_coordinate.loc[i + 1, "lon"] = new_coordinate.loc[i, "lon"]
        #         new_coordinate.loc[i + 1, "lat"] = new_coordinate.loc[i, "lat"]
        #         continue
        #     array_position1 = np.where(distance_matrix <= a)[0][-1]
        #     if array_position1 == (len(distance_matrix) - 1):
        #         array_position2 = array_position1
        #         array_position1 = array_position1 - 1
        #         remained_distance = a - distance_matrix[array_position1]
        #         original_distance = distance_matrix[array_position2] - distance_matrix[array_position1]
        #         position = cal_new_coordinate(item.loc[array_position1, "lon"], item.loc[array_position1, "lat"],
        #                                       item.loc[array_position2, "lon"], item.loc[array_position2, "lat"],
        #                                       original_distance, remained_distance)
        #         new_coordinate.loc[i + 1, "lon"] = position[0]
        #         new_coordinate.loc[i + 1, "lat"] = position[1]
        #     else:
        #         array_position2 = np.where(distance_matrix >= a)[0][0]
        #         remained_distance = a - distance_matrix[array_position1]
        #         original_distance = distance_matrix[array_position2] - distance_matrix[array_position1]
        #         position = cal_new_coordinate(item.loc[array_position1, "lon"], item.loc[array_position1, "lat"],
        #                                       item.loc[array_position2, "lon"], item.loc[array_position2, "lat"],
        #                                       original_distance, remained_distance)
        #         new_coordinate.loc[i + 1, "lon"] = position[0]
        #         new_coordinate.loc[i + 1, "lat"] = position[1]

        # smoothed_acceleration = [smoothed_original_velocity[i + 1] - smoothed_original_velocity[i] for i in range(len(smoothed_original_velocity) - 1)]
        # smoothed_acceleration = np.array(smoothed_acceleration)
        #
        # brake_distance = pd.DataFrame(columns=["distance", "road_type"])
        # brake_distance_i = 0
        # for i in range(len(smoothed_acceleration)):
        #     if smoothed_acceleration[i] > brake_deceleration_threshold:
        #         brake_distance.loc[brake_distance_i, "distance"] = smoothed_original_velocity[i]
        #         brake_distance.loc["road_type"] = item.loc[i, "road_type"]

    print("complete: {:.4%}".format(j/len(trajectory_file_path)))
    return 0


if __name__ == "__main__":
    trajectory_file_path = []
    for root, dirs, files in os.walk(file_dir):
        for filename in files:
            trajectory_file_path.append(os.path.join(root, filename))
    trajectory_file_path_list = []
    for i in range(len(trajectory_file_path)):
        trajectory_file_path_list.append((trajectory_file_path, i))
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(number_of_parrel) as pool:
        pool.starmap(main_function, trajectory_file_path_list)