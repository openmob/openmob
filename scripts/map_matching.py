import osmnx as ox
import folium
import networkx as nx
import random
from osmnx import utils_graph
from osmnx import distance
import pandas as pd
import datetime
import numpy as np


custom_filter = '["highway"~"motorway|trunk|primary|secondary|tertiary"]'
ox.config(use_cache=True)
# 35.79,35.55,139.62,139.89
# 36.86,34.89,138.75,141.00
g = ox.graph_from_bbox(36.86,34.89,138.75,141.00, simplify=True, network_type='drive',
                       custom_filter=custom_filter)  # north, south, east, west,
g = ox.utils_graph.get_largest_component(g,strongly=True)
print('地图载入完成')
def calc_traj():
    data = pd.read_csv('all_valid_final_1.csv')

    day = data.iloc[:,0]
    uid = data.iloc[:,1]
    lat = data.iloc[:,2]
    long = data.iloc[:,3]
    str_time = data.iloc[:,4]

    prev_day = day[0]
    prev_uid = int(uid[0])
    prev_lat = float(lat[0])
    prev_long = float(long[0])
    prev_str_time = str_time[0]

    full_route = []
    for i in range(len(data)):
        curr_day = day[i]
        curr_uid = int(uid[i])
        curr_lat = float(lat[i])
        curr_long = float(long[i])
        curr_str_time = str_time[i]
        if prev_uid == curr_uid:
            source = ox.distance.nearest_nodes(g, prev_long, prev_lat)
            target = ox.distance.nearest_nodes(g, curr_long, curr_lat)
            route = nx.shortest_path(g, source, target)
            t_start = datetime.datetime.strptime(prev_str_time, "%Y-%m-%d %H:%M:%S")
            t_end = datetime.datetime.strptime(curr_str_time, "%Y-%m-%d %H:%M:%S")
            sub = (t_end-t_start)/len(route)
            t0 = t_start
            for item in route:
                full_route.append([curr_day,curr_uid,
                                   g.nodes[item]['y'], g.nodes[item]['x'],
                                  (t0).strftime("%Y-%m-%d %H:%M:%S")])
                t0 =t0 + sub
            prev_day = curr_day
            #prev_uid = curr_uid
            prev_lat = curr_lat
            prev_long = curr_long
            prev_str_time = curr_str_time
        else:
            print(curr_uid)
            df = pd.DataFrame(data=full_route)
            df.to_csv('sp_traj/'+str(prev_uid)+'.csv', header=False, index=False)
            full_route = []
            prev_day = curr_day
            prev_uid = curr_uid
            prev_lat = curr_lat
            prev_long = curr_long
            prev_str_time = curr_str_time


calc_traj()

