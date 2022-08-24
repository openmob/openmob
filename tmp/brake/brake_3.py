import pywt
import natsort
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

def calculate_dis(O_lat, O_lon, D_lat, D_lon):
    a = np.radians(np.asarray(O_lon, dtype=np.float64) - np.asarray(D_lon, dtype=np.float64))
    b = np.radians(np.asarray(O_lat, dtype=np.float64) - np.asarray(D_lat, dtype=np.float64))
    dis = 2 * R * np.arcsin(np.sqrt(np.sin(b/2) ** 2 + np.cos(np.radians(np.asarray(O_lat, dtype=np.float64))) * np.cos(np.radians(np.asarray(D_lat, dtype=np.float64))) * np.sin(a/2) ** 2))
    return dis

def cal_new_coordinate(position1_x, position1_y, position2_x, position2_y, original_distance, velocity):
    if velocity == 0:
        return (position1_x, position1_y)
    if (position2_x >= position1_x) and (position2_y >= position1_y):
        return (position1_x + abs(position2_x - position1_x)/original_distance * velocity, position1_y + abs(position2_y - position1_y)/original_distance * velocity)
    elif (position2_x >= position1_x) and (position2_y <= position1_y):
        return (position1_x + abs(position2_x - position1_x)/original_distance * velocity, position1_y - abs(position2_y - position1_y)/original_distance * velocity)
    elif (position2_x <= position1_x) and (position2_y >= position1_y):
        return (position1_x - abs(position2_x - position1_x)/original_distance * velocity, position1_y + abs(position2_y - position1_y)/original_distance * velocity)
    elif (position2_x <= position1_x) and (position2_y <= position1_y):
        return (position1_x - abs(position2_x - position1_x)/original_distance * velocity, position1_y - abs(position2_y - position1_y)/original_distance * velocity)



file = "D:/paper/brake/data/all_trajectory.csv"
figure_save_path = "D:/paper/brake/figure/"
R = 6371008
threshold = 0.1
maximum_acceleration = 6
if __name__=="__main__":
    data = pd.read_csv(file, low_memory=False)
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
        raw_velocity_list = calculate_dis(coordinate_df["from_lat"], coordinate_df["from_lon"], coordinate_df["to_lat"],
                                        coordinate_df["to_lon"])

        distance_matrix = np.zeros([len(item)])
        distance_matrix[0] = 0
        for i in range(len(raw_velocity_list)):
            distance_matrix[i + 1] = np.sum(raw_velocity_list[:(i + 1)])

        raw_acceleration_list = [raw_velocity_list[i + 1] - raw_velocity_list[i] for i in range(len(raw_velocity_list)-1)]
        raw_acceleration_list = np.array(raw_acceleration_list)
        velocity_list_1 = np.zeros(len(raw_velocity_list))
        velocity_list_1[0] = raw_velocity_list[0]
        #去噪
        for i in range(len(raw_acceleration_list)):
            if abs(raw_acceleration_list[i]) <= 6:
                velocity_list_1[i + 1] = raw_velocity_list[i + 1]
            else:
                if i == (len(raw_acceleration_list) - 2):
                    velocity_list_1[i + 1] = velocity_list_1[i] + raw_acceleration_list[i - 1]
                else:
                    velocity_list_1[i + 1] = (velocity_list_1[i] + velocity_list_1[i + 2])/2 #差值补空

        x = np.arange(0, len(velocity_list_1))

        # 小波滤波
        # wave = pywt.Wavelet('haar')
        # max_level = pywt.dwt_max_level(len(velocity_list_1), wave.dec_len)
        # coeffs = pywt.wavedec(velocity_list_1, 'haar', level=max_level)
        # for i in range(1, len(coeffs)):
        #     coeffs[i] = pywt.threshold(coeffs[i], threshold * np.max(coeffs[i]))
        # velocity_list_2 = pywt.waverec(coeffs, 'haar')
        # velocity_list_2 = velocity_list_2[0: len(velocity_list_2) - 1]

        plt.figure()
        plt.plot(x, raw_velocity_list)
        plt.xlabel('time (s)')
        plt.ylabel('velocity(m/s)')
        plt.title("Raw velocity_" + str(driver_number))
        plt.savefig(figure_save_path + "Raw velocity_"+str(driver_number)+".jpg")

        plt.figure()
        plt.plot(x, velocity_list_1)
        plt.xlabel('time (s)')
        plt.ylabel('velocity(m/s)')
        plt.title("no outlier velocity_" + str(driver_number))
        plt.savefig(figure_save_path + "No outlier velocity_"+str(driver_number)+".jpg")
        # plt.subplot(2, 1, 2)
        # plt.plot(x, velocity_list_2)
        # plt.xlabel('time (s)')
        # plt.ylabel('velocity(m/s)')
        # plt.title("De-noised velocity_" + str(driver_number))
        # plt.tight_layout()
        # plt.savefig(figure_save_path + "wavelet_contrast"+str(driver_number)+".jpg")
        # plt.show()


        #计算X,Y上的速度分量
        # velocity_list_2_x = []
        # velocity_list_2_y = []
        # for i in range(len(iter_item) - 1):
        #     if (iter_item.loc[i + 1, "lon"] == iter_item.loc[i, "lon"]) and (iter_item.loc[i + 1, "lat"] == iter_item.loc[i, "lat"]):
        #         velocity_list_2_x.append(0)
        #         velocity_list_2_y.append(0)
        #     else:
        #         a = abs((iter_item.loc[i + 1, "lon"] - iter_item.loc[i, "lon"]))/np.sqrt(((iter_item.loc[i + 1, "lon"] - iter_item.loc[i, "lon"])**2 + (iter_item.loc[i + 1, "lat"] - iter_item.loc[i, "lat"])**2)) * velocity_list_2[i]
        #         velocity_list_2_x.append(a)
        #         b = abs((iter_item.loc[i + 1, "lat"] - iter_item.loc[i, "lat"])) / np.sqrt(((iter_item.loc[i + 1, "lon"] - iter_item.loc[i, "lon"]) ** 2 + (iter_item.loc[i + 1, "lat"] - iter_item.loc[i, "lat"]) ** 2)) * velocity_list_2[i]
        #         velocity_list_2_y.append(b)
        # velocity_list_2_x = np.asarray(velocity_list_2_x)
        # velocity_list_2_y = np.asarray(velocity_list_2_y)

        #
        # new_item = []
        # new_item.append((item.loc[0, "lon"], item.loc[0, "lat"]))
        # for i in range(len(velocity_list_2)):
        #     distance_array = distance_array[i, i:]
        #     new_item.append(cal_new_coordinate(item.loc[i, "lon"], item.loc[i, "lat"],
        #                     item.loc[i + 1, "lon"], item.loc[i + 1, "lat"],
        #                     velocity_list_1[i], velocity_list_2[i], distance_array))
        # new_item = np.asarray(new_item)
        #
        # velocity_list_2_x = np.append(velocity_list_2_x, 0)
        # velocity_list_2_y = np.append(velocity_list_2_y, 0)
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
        # (filtered_state_means, filtered_state_covariances) = kf.filter(state_variable)

        x = np.arange(0, len(smoothed_state_means))
        smoothed_original_velocity = smoothed_state_means[:, 0]
        # smoothed_wavelet_velocity = smoothed_state_means[:, 1]
        # filtered_velocity = np.sqrt((filtered_state_means[:, 0] ** 2 + filtered_state_means[:, 1] ** 2))

        # plt.figure(1)
        # plt.plot(x, filtered_velocity, 'bo')
        # plt.show()
        # x, velocity_list_2, 'b--',
        plt.figure()
        plt.plot(x, smoothed_original_velocity, 'r')
        plt.title("smoothed velocity_" + str(driver_number))
        plt.savefig(figure_save_path + "smoothed velocity_"+str(driver_number)+".jpg")

        smoothed_acceleration = np.zeros(len(smoothed_original_velocity))


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
                position = cal_new_coordinate(item.loc[array_position1, "lon"], item.loc[array_position1, "lat"],
                                              item.loc[array_position2, "lon"], item.loc[array_position2, "lat"],
                                              original_distance, remained_distance)
                new_coordinate.loc[i + 1, "lon"] = position[0]
                new_coordinate.loc[i + 1, "lat"] = position[1]
            else:
                array_position2 = np.where(distance_matrix >= a)[0][0]
                remained_distance = a - distance_matrix[array_position1]
                original_distance = distance_matrix[array_position2] - distance_matrix[array_position1]
                position = cal_new_coordinate(item.loc[array_position1, "lon"], item.loc[array_position1, "lat"],
                                              item.loc[array_position2, "lon"], item.loc[array_position2, "lat"],
                                              original_distance, remained_distance)
                new_coordinate.loc[i + 1, "lon"] = position[0]
                new_coordinate.loc[i + 1, "lat"] = position[1]
        new_coordinate.to_csv("D:/paper/brake/filtered"+str(driver_number)+".csv")
        print("{:.4%}".format(driver_number/len(grouped_data)))
        driver_number += 1


    # coordinate = []
    # for i in range(len(item)):
    #     coordinate.append((item.loc[i, "lon"], item.loc[i, "lat"]))
    # coordinate = np.asarray(coordinate)
    # wave = pywt.Wavelet('db12')
    # max_level = pywt.dwt_max_level(len(coordinate), wave.dec_len)
    # coeffs = pywt.wavedec2(coordinate, 'db12', level=max_level)
    # print(coeffs[1][2])
    # for i in range(1, len(coeffs)):
    #     for k in range(len(coeffs[i][2])):
    #         coeffs[i][2][k] = pywt.threshold(coeff
    #             plt.plot(x, raw_velocity_list)s[i][2][k], threshold * np.max(coeffs[i][2]), mode='soft')
    #
    # new_coordinate = pywt.waverec2(coeffs, 'db12')  # 将信号进行小波重构
    # new_coordinate_df = pd.DataFrame(columns=["from_lon", "from_lat", "to_lon", "to_lat"])
    # for i in range(len(new_coordinate) - 1):
    #     new_coordinate_df.loc[i, "from_lon"] = new_coordinate[i][0]
    #     new_coordinate_df.loc[i, "from_lat"] = new_coordinate[i][1]
    #     new_coordinate_df.loc[i, "to_lon"] = new_coordinate[i + 1][0]
    #     new_coordinate_df.loc[i, "to_lat"] = new_coordinate[i + 1][1]
    # dis_list_2 = calculate_dis(new_coordinate_df["from_lat"], new_coordinate_df["from_lon"],
    #                            new_coordinate_df["to_lat"],
    #                            new_coordinate_df["to_lon"])