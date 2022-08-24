import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import multiprocessing
import tqdm
import pickle
import numpy as np
working_dir = r"D:\paper\parking\1C3P"
save_dir = working_dir + r"\optimization\yongdi_travel_count"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

processed_file = []
for root, dirs, files in os.walk(save_dir):
    for filename in files:
        processed_file.append(filename)

dir = working_dir + r"\trajectory_in_study_area"
path = []
for root, dirs, files in os.walk(dir):
    file_to_do = list(set(files).difference(set(processed_file)))
    for filename in file_to_do:
        path.append(os.path.join(root, filename))

yongdi = gpd.GeoDataFrame.from_file(working_dir + r"\mapping\yongdi_new.shp", encoding="UTF-8")

yongdi_a = yongdi.copy()
yongdi_a = yongdi_a.to_crs(epsg=3395)

def geo_function(x):
    matched_yongdi = yongdi[yongdi["geometry"].contains(x.at["geometry"])]
    if len(matched_yongdi) == 0:
        yongdi_a["distance"] = yongdi_a["geometry"].apply(lambda y: y.boundary.distance(x.at["geometry"]))
        area_code = yongdi_a.loc[yongdi_a["distance"].idxmin(), "area_code"]
    else:
        area_code = matched_yongdi.loc[matched_yongdi.index[0], "area_code"]
    return area_code

def function_1(file):
    data = pd.read_csv(file, index_col=[0])
    data["geometry"] = data["geometry"].apply(wkt.loads)
    data = gpd.GeoDataFrame(data, geometry="geometry")
    data.crs = 4326
    data = data.to_crs(epsg=3395)
    result = pd.DataFrame(columns=["trip_number", "area_code", "trip_quantity"])
    try:
        data["area_code"] = data.apply(geo_function, axis=1)
    except:
        print(file)
        return
    z = 0
    a = data.groupby("trip_number").groups
    b = list(a)
    for i in range(len(b)):
        data_1 = data.loc[a[b[i]], :]
        area_code_list = list(data_1["area_code"])
        area_code_list = list(set(area_code_list))
        for j in range(len(area_code_list)):
            result.loc[z, "trip_number"] = data_1.loc[data_1.index[0], "trip_number"]
            result.loc[z, "area_code"] = area_code_list[j]
            result.loc[z, "trip_quantity"] = 1
            z += 1
    if len(result) == 0:
        return
    name = file.split("\\")[-1]
    result.to_csv(save_dir + "/" + name)
    return


def read_function(file):
    data = pd.read_csv(file, index_col=[0])
    return data

if __name__=="__main__":
    # function_1(r"D:\paper\parking\1C3P\trajectory_in_study_area\00401621.csv")
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        list(tqdm.tqdm(p.imap_unordered(function_1, path), total=len(path)))