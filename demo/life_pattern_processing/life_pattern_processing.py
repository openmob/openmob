from openmob.scripts.life_pattern_processor import *
import os


if __name__ == '__main__':
    print(os.getcwd())
    lpp = LifePatternProcessor(initialize=True,
                               raw_gps_folder='../../datasets/separate_trip/',
                               map_file='../../datasets/gis/greater_tokyo_area_dissolve.shp',
                               )