import glob
import multiprocessing
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import tqdm
from shapely.geometry import LineString

tqdm.tqdm.pandas()


def pa_assign(file):
    #     print(time.ctime(), file, '\n')
    results_file = '/mnt/efs/huang/data/labeled_points/labeled_stay_points_{}.csv'.format(file.split('/')[-1][:6])

    sample = pd.read_csv(file)
    sample = sample.drop('Unnamed: 0', axis=1)
    sample = sample[sample.lon != 'lon']

    #     sample_all = pd.concat([sample_all, sample], axis=0)

    sample['lon'] = sample.lon.astype(float)
    sample['lat'] = sample.lat.astype(float)
    #     sample = pd.merge(sample_all, label, how='left', on='M2M_MODEL_CD')

    sample_geo = gpd.GeoDataFrame(sample, crs='epsg:4326', geometry=gpd.points_from_xy(sample.lon, sample.lat))

    for k in range(len(gj['features'])):
        #         print(time.ctime(), k+1, len(gj['features']))
        pa_name = gj['features'][k]['prorperties']['name']
        #         pa_location = df.loc[i, 'geom']
        pa_shp = shapely.geometry.shape(gj['features'][k]['geometry'])

        pa_shp_geo = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[pa_shp])
        #         pa_shp_geo.crs = 'epsg:4326'
        #         pa_shp_geo = pa_shp_geo.to_crs(epsg=3763)
        #         buffer = pa_shp_geo.buffer(500, cap_style=3)

        buffered = sample_geo.loc[sample_geo.geometry.within(pa_shp_geo.loc[0, 'geometry']), :]
        buffered['pa_name'] = pa_name

        buffered.to_csv(results_file, mode='a')


#     return buffered

if __name__ == '__main__':
    filelist = glob.glob('/mnt/efs/huang/data/stay_points/*.csv')
    filelist = np.sort(filelist)

    import warnings

    warnings.filterwarnings('ignore')

    start = time.time()
    with multiprocessing.Pool(12) as pool:
        pool.map(pa_assign, filelist)

    print(time.time() - start)
