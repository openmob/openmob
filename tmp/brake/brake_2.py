import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

def interpo(item, key):
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
                interpolated_item.loc[interpolated_item_i, "time"] = interpolated_item.loc[interpolated_item_i - 1, "time"] + datetime.timedelta(seconds=1)
                interpolated_item.loc[interpolated_item_i, "lon"] = interpolation_array[j, 0]
                interpolated_item.loc[interpolated_item_i, "lat"] = interpolation_array[j, 1]
                interpolated_item.loc[interpolated_item_i, "time_interval"] = 1
                interpolated_item.loc[interpolated_item_i, "distance"] = 0
                interpolated_item.loc[interpolated_item_i, "vehicle_type"] = item.loc[0, "vehicle_type"]
                interpolated_item.loc[interpolated_item_i, "interpolation"] = 1
                interpolated_item_i += 1
    return interpolated_item

def duankai_interpo(item, dijige, key):
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
                interpolated_item.loc[interpolated_item_i, "time"] = interpolated_item.loc[interpolated_item_i - 1, "time"] + datetime.timedelta(seconds=1)
                interpolated_item.loc[interpolated_item_i, "lon"] = interpolation_array[j, 0]
                interpolated_item.loc[interpolated_item_i, "lat"] = interpolation_array[j, 1]
                interpolated_item.loc[interpolated_item_i, "time_interval"] = 1
                interpolated_item.loc[interpolated_item_i, "distance"] = 0
                interpolated_item.loc[interpolated_item_i, "vehicle_type"] = item.loc[0, "vehicle_type"]
                interpolated_item.loc[interpolated_item_i, "interpolation"] = 1
                interpolated_item_i += 1
    return interpolated_item

file_dir = "D:/paper/brake/data/all_trajectory.csv"
if __name__=="__main__":
    all_trajectory = pd.read_csv(file_dir, index_col=[0], low_memory=False)
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
            interpolated_item = interpo(item, key)
            # interpolation_x = np.array([item.loc[i - 1, "lon"], item.loc[i - 1, "lat"]])
            # interpolation_y = np.array([item.loc[i - 1, "lon"], item.loc[i - 1, "lat"]])
            interpolated_item.to_csv("D:/paper/brake/data/interpolation/interpolated_trajectory_" + interpolated_item.loc[0, "routine_ID"] + ".csv")
        else:
            for i in range(len(duandian)):
                if i == 0:
                    duankai_item = item.loc[:(duandian.index[i] - 1), :]
                else:
                    duankai_item = item.loc[duandian.index[i - 1]:(duandian.index[i] - 1), :]
                if len(duankai_item) > 1:
                    interpolated_item = duankai_interpo(duankai_item, i, key)
                    interpolated_item.to_csv("D:/paper/brake/data/interpolation/interpolated_trajectory_" + interpolated_item.loc[0, "routine_ID"] + ".csv")
            duankai_item = duankai_item = item.loc[duandian.index[-1]:, :]
            if len(duankai_item) > 1:
                interpolated_item = duankai_interpo(duankai_item, (i+1), key)
                interpolated_item.to_csv("D:/paper/brake/data/interpolation/interpolated_trajectory_" + interpolated_item.loc[0, "routine_ID"] + ".csv")
        jishu += 1
        print("{:.4%}".format(jishu / len(grouped_all_trajectory)))