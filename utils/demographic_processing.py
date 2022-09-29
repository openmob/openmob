# 1.Initial Lines
# !/usr/bin/env python
# -*- coding: utf-8 -*-

# 2.Note for this file.
"""This file is for mobaku pre-processing"""
__author__ = 'Li Peiran'

# 3.Import the modules.
import os

import geopandas as gpd
import jismesh.utils as ju
import pandas as pd
from pandas import DataFrame
from shapely import geometry
import minitools


def createPolygon(meshcode):
    """
    Create polygon for corresponding mesh code

    Args:
        meshcode from jmesh
    Retruns:
        created polygon
    """
    lat1, lon1 = ju.to_meshpoint(meshcode, 0, 0)
    lat2, lon2 = ju.to_meshpoint(meshcode, 1, 1)
    polygon = geometry.Polygon([(lon1, lat1), (lon1, lat2), (lon2, lat2), (lon2, lat1)])
    return polygon


def transMobaku2Shape(MOBAKU_FOLDER_PATH, OUTPUT_FOLDER_PATH=''):
    """
    Load Mobaku CSV file to shapefile

    Args:
        Mobaku data folder path
        1. Folder must contain: '01_総数.csv'
        '02_性年代(10歳階).csv'
        2. and mobaku data must
        be follwing structure:
        date/day_of_week/time/area/residence/age/gender/population
        20191201/17/0/533925454/-1/15/1/102
    Returns:
        population.shp: population shapefile with WGS-84
        demographic.shp: age/gender shapefile with WGS-84
    """

    POP_PATH = MOBAKU_FOLDER_PATH + '01_総数.csv'
    AGE_PATH = MOBAKU_FOLDER_PATH + '02_性年代(10歳階).csv'
    print('Loading the data...')
    pop_df = pd.read_csv(POP_PATH)
    demographic_df = pd.read_csv(AGE_PATH)

    # Load and save as shapefile
    pop_df['geometry'] = pop_df.apply(lambda row: createPolygon(row.area), axis=1)
    pop_geo_df = gpd.GeoDataFrame(data=pop_df, crs='EPSG:4326', geometry='geometry')
    print('Export the population.shp...')
    pop_geo_df.to_file(OUTPUT_FOLDER_PATH + 'population.shp')

    demographic_df['geometry'] = demographic_df.apply(lambda row: createPolygon(row.area), axis=1)
    demographic_geo_df = gpd.GeoDataFrame(data=demographic_df, crs='EPSG:4326', geometry='geometry')
    print('Export the demographic.shp...')
    demographic_geo_df.to_file(OUTPUT_FOLDER_PATH + 'demographic.shp')

    return 1


def transMobaku2PKL(MOBAKU_DEMOGRAPHIC_PATH, OUTPUT_FOLDER_PATH=''):
    """
    Load Mobaku CSV file to pkl file (preparing for fast labeling)

    Args:
        MOBAKU_DEMOGRAPHIC_PATH: Mobaku data 02_性年代(10歳階) path
    Returns:
        output 2 folders contains corresponding population
        of a certain mesh and time:
        1. by_absolute_date
        2. by_day_of_week
    Note:
        mobaku data must be follwing structure:
        date/day_of_week/time/area/residence/age/gender/population
        20191201/17/0/533925454/-1/15/1/102
        OUTPUT_FOLDER_PATH:
        The target folder to save
    """

    demographic_df = pd.read_csv(MOBAKU_DEMOGRAPHIC_PATH)
    demographic_df = demographic_df.drop(['residence'], axis=1)

    # Check the save folders
    minitools.if_folder_exist_then_create(OUTPUT_FOLDER_PATH + 'by_absolute_date/')
    minitools.if_folder_exist_then_create(OUTPUT_FOLDER_PATH + 'by_day_of_week/')

    # With absolute date
    for key, item in demographic_df.groupby(['age', 'gender']):
        temp_dict = {(x, y, z): value for (x, y, z, value) in
                     item[['date', 'time', 'area', 'population']].values}
        minitools.save_pkl(temp_dict, OUTPUT_FOLDER_PATH + 'by_absolute_date/' + str(key))

    # With the day_of_week
    for key, item in demographic_df.groupby(['age', 'gender']):
        temp_dict = {(x, y, z): value for (x, y, z, value) in
                     item[['day_of_week', 'time', 'area', 'population']].values}
        minitools.save_pkl(temp_dict, OUTPUT_FOLDER_PATH + 'by_day_of_week/' + str(key))


def getSAfromMobakuPkl(MOBAKU_PKL_FOLDER_PATH, load_mode='by_day_of_week'):
    """
    Merge the pkl, return the demographic dataframe

    Args:
        MOBAKU_PKL_FOLDER_PATH:
        The folder contains population list of different age/genders(.pkl)
        load_mode:
        Load 'by_day_of_week' or 'by_absolute_date'
    Returns:
        The SA matrix
    """
    PKL_PATH = MOBAKU_PKL_FOLDER_PATH + '/' + load_mode
    folder_list = []
    PKL_list = []
    minitools.get_file_path(PKL_PATH, PKL_list, folder_list, '.pkl')
    demographic_df_list = []
    for file in PKL_list:
        temp_pkl = minitools.load_pkl(file)
        demographic_df_list.append(temp_pkl)
    demographic_df = DataFrame(demographic_df_list).T
    demographic_df.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any NaN values
    return demographic_df


def generateCsvSAfromMobakuPkl(MOBAKU_PKL_FOLDER_PATH, load_mode='by_day_of_week'):
    """
    Merge the pkl, return the demographic dataframe,and save as .csv file

    Args:
        MOBAKU_PKL_FOLDER_PATH:
        The folder contains population list of different age/genders(.pkl)
        load_mode:
        Load 'by_day_of_week' or 'by_absolute_date'
    Returns:
        1
    """
    PKL_PATH = (os.path.join(MOBAKU_PKL_FOLDER_PATH, load_mode))
    folder_list = []
    PKL_list = []
    minitools.get_file_path(PKL_PATH, PKL_list, folder_list, '.pkl')
    demographic_df_list = []
    for file in PKL_list:
        temp_pkl = minitools.load_pkl(file)
        demographic_df_list.append(temp_pkl)
    demographic_df = DataFrame(demographic_df_list).T
    demographic_df.dropna(axis=0, how='any', inplace=True)  # drop all rows that have any NaN values
    demographic_df.to_csv(os.path.join(PKL_PATH, 'SA_GT_df.csv'))
    return 1


# 6. Define the main function (if exsists)
if __name__ == '__main__':
    MOBAKU_FOLDER_PATH = r'C:\Users\lipeiran\OneDrive - The University of Tokyo\Mobility_Behaviour_Analysis_COVID19\00_cencus_data\\'
    MOBAKU_PKL_FOLDER_PATH = r'C:\Users\lipeiran\OneDrive - The University of Tokyo\Python_Script\Fast_Labeling\FastLabelingProject_for_System\assets\mobaku_demographic_pkl/'
    MOBAKU_DEMOGRAPHIC_PATH = MOBAKU_FOLDER_PATH + '02_性年代(10歳階).csv'
    # transMobaku2Shape(MOBAKU_FOLDER_PATH,OUTPUT_FOLDER_PATH)
    # transMobaku2PKL(MOBAKU_DEMOGRAPHIC_PATH, MOBAKU_PKL_FOLDER_PATH)
    getSAfromMobakuPkl(MOBAKU_PKL_FOLDER_PATH, load_mode='by_day_of_week')
    generateCsvSAfromMobakuPkl(MOBAKU_PKL_FOLDER_PATH, load_mode='by_day_of_week')
