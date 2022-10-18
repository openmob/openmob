import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import wkt
import multiprocessing
import tqdm
import pickle
import numpy as np
import haversine
import datetime
import re



class hydrogen_station:
    def __init__(self,
                 working_dir = r"D:\paper\parking\1C3P",
                 docomo_2011_dir = r"H:\NTT_Docomo_ZDC_2011",
                 initialize = False
                 ):

        self.working_dir = working_dir
        self.save_dir1 = self.working_dir + r"\vehicle_trajectory/"
        self.save_dir2 = self.working_dir + r"\optimization\yongdi_travel_count"
        self.docomo_2011_dir = docomo_2011_dir
        self.dir = self.working_dir + r"\trajectory_in_study_area"
        self.initialize = initialize

        if self.initialize:
            self.create_folder()
            self.processed_file1 = []
            self.processed_file2 = []
            self.processed_file()
            self.path_2011 = self.generate_to_do_filepath(filepath= self.docomo_2011_dir, processed_file = self.processed_file1)
            self.path_2 = self.generate_to_do_filepath(filepath = self.dir, processed_file = self.processed_file2)
            with multiprocessing.Pool(24) as p:
                list(tqdm.tqdm(p.imap_unordered(self.function_1_1, self.path_2011), total=len(self.path_2011)))
            with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
                list(tqdm.tqdm(p.imap_unordered(self.function_2_1, self.path_2), total=len(self.path_2)))



    def create_folder(self):
        if not os.path.exists(self.save_dir1):
            os.makedirs(self.save_dir1)
        if not os.path.exists(self.save_dir2):
            os.makedirs(self.save_dir2)
        return

    def processed_file(self):
        for root, dirs, files in os.walk(self.save_dir1):
            for filename in files:
                self.processed_file1.append(root, filename)
        for root, dirs, files in os.walk(self.save_dir2):
            for filename in files:
                self.processed_file2.append(root, filename)
        return

    def generate_to_do_filepath(self, filepath, processed_file):
        to_do_filepath = []
        for root, dirs, files in os.walk(filepath):
            file_to_do = list(set(files).difference(set(processed_file)))
            for filename in file_to_do:
                to_do_filepath.append(os.path.join(root, filename))
        return to_do_filepath

    def function_1_1(self, path_2011):
        file = [x for x in path_2011 if os.stat(x).st_size != 0]

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
        location.to_csv(self.save_dir1 + b)
        return

    def geo_function(self, x):
        yongdi = gpd.GeoDataFrame.from_file(self.working_dir + r"\mapping\yongdi_new.shp", encoding="UTF-8")

        yongdi_a = yongdi.copy()
        yongdi_a = yongdi_a.to_crs(epsg=3395)
        matched_yongdi = yongdi[yongdi["geometry"].contains(x.at["geometry"])]
        if len(matched_yongdi) == 0:
            yongdi_a["distance"] = yongdi_a["geometry"].apply(lambda y: y.boundary.distance(x.at["geometry"]))
            area_code = yongdi_a.loc[yongdi_a["distance"].idxmin(), "area_code"]
        else:
            area_code = matched_yongdi.loc[matched_yongdi.index[0], "area_code"]
        return area_code

    def function_2_1(self, file):
        data = pd.read_csv(file, index_col=[0])
        data["geometry"] = data["geometry"].apply(wkt.loads)
        data = gpd.GeoDataFrame(data, geometry="geometry")
        data.crs = 4326
        data = data.to_crs(epsg=3395)
        result = pd.DataFrame(columns=["trip_number", "area_code", "trip_quantity"])
        try:
            data["area_code"] = data.apply(self.geo_function, axis=1)
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
        result.to_csv(self.save_dir2 + "/" + name)
        return

if __name__ == '__main__':
    hydrogen_station(initialize = True)