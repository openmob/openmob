import os
import pandas as pd
import haversine
import datetime
# import pickle4reducer
import multiprocessing
# ctx = multiprocessing.get_context()
# ctx.reducer = pickle4reducer.Pickle4Reducer()
import tqdm
import re
working_dir = r"D:\paper\parking\1C3P"
save_dir = working_dir + r"\vehicle_trajectory/"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

processed_file = []
for root, dirs, files in os.walk(save_dir):
    for filename in files:
        processed_file.append(root, filename)

docomo_2011_dir = r"H:\NTT_Docomo_ZDC_2011"
path_2011 = []
for root, dirs, files in os.walk(docomo_2011_dir):
    file_to_do = list(set(files).difference(set(processed_file)))
    for filename in file_to_do:
        path_2011.append(os.path.join(root, filename))
path_2011 = [x for x in path_2011 if os.stat(x).st_size != 0]

def function_1(file):
    b = file.split("\\")[-1]
    data = pd.read_csv(file, header=None, usecols=[1, 3, 10, 11, 16])
    data.columns = ["date", "mode", "start_time", "end_time", "trajectory"]
    data = data[data["mode"] == "CAR"]
    if len(data) == 0:
        return
    location = pd.DataFrame(columns=["time", "latitude", "longitude", "mode", "trip_number"])
    z = 0
    trip_num = 0
    result = []
    for i in range(len(data)):
        trajectory1 = data.loc[data.index[i], "trajectory"].split(";")
        for j in range(len(trajectory1)):
            a = trajectory1[j].split("|")
            location.loc[z, "time"] = a[1]
            location.loc[z, "latitude"] = a[2]
            location.loc[z, "longitude"] = a[3]
            location.loc[z, "mode"] = data.loc[data.index[i], "mode"]
            location.loc[z, "trip_number"] = trip_num
            z += 1
            if z > 5000:
                result.append(location)
                location = pd.DataFrame(columns=["time", "latitude", "longitude", "mode", "trip_number"])
                z = 0
        trip_num += 1
    result.append(location)
    location = pd.concat(result, ignore_index=True, sort=False)
    location.to_csv(save_dir + b)
    return

if __name__=="__main__":
    # function_1(path_2011[0])
    with multiprocessing.Pool(24) as p:
        list(tqdm.tqdm(p.imap_unordered(function_1, path_2011), total=len(path_2011)))
    # location = pd.concat(location, ignore_index=True, sort=False)
    # location.to_csv(r'D:\paper\travel_mode\all_location.csv')

    # with multiprocessing.Pool(12) as p:
    #     list(tqdm.tqdm(p.imap_unordered(function_1, processed_file), total=len(processed_file)))