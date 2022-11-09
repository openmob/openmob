
from decimal import *
import jismesh.utils as ju
import numpy as np
import pandas as pd


class Point(object):
    def __init__(self, place, lat=None, lon=None):
        getcontext().prec = 8
        print('Selected point:', place)
        if place == 'tokyo_station':
            # 东京站
            # order: 2773; index: (34, 53); meshcode: '533946113'
            # grid_gps: [[35.684, 139.765], [35.68, 139.77]]
            self.lat = Decimal('35.681111')
            self.lon = Decimal('139.766944')
        elif place == 'shinjuku_station':
            # order: 2600; index: (32, 40); meshcode: '533945263'
            # grid_gps: [[35.692, 139.7], [35.688, 139.705]]
            self.lat = Decimal('35.690281')
            self.lon = Decimal('139.700561')
        elif place == 'ikebukuro_station':
            # order: 1802; index: (22, 42); meshcode: '533945773'
            # grid_gps: [[35.732, 139.71], [35.728, 139.715]]
            self.lat = Decimal('35.730278')
            self.lon = Decimal('139.711389')
        elif place == 'shibuya_station':
            # order: 3240; index: (40, 40); meshcode: '533935863'
            # grid_gps: [[35.66, 139.7], [35.656, 139.705]]
            self.lat = Decimal('35.658611')
            self.lon = Decimal('139.701389')
        elif place == 'shimbashi_station':
            # order: 3091; index: (38, 51); meshcode: '533936904'
            # grid_gps: [[35.668, 139.755], [35.664, 139.76]]
            self.lat = Decimal('35.666389')
            self.lon = Decimal('139.7584')
        elif place == 'akihabara_station':
            # order: 2454; index: (30, 54); meshcode: '533946314'
            # grid_gps: [[35.7, 139.77], [35.696, 139.775]]
            self.lat = Decimal('35.698333')
            self.lon = Decimal('139.773311')
        elif place == 'custom':
            self.lat = Decimal(lat)
            self.lon = Decimal(lon)
        else:
            print('invalid region name...')
            assert False


class Region(object):
    def __init__(self, city, region, size):
        getcontext().prec = 8
        if region == 'tokyo':
            # 4 * 3
            # 32-35, 52-54
            # station (34, 53)
            self.minLat = Decimal('35.676')
            self.maxLat = Decimal('35.692')
            self.minLon = Decimal('139.76')
            self.maxLon = Decimal('139.775')
        elif region == 'shinjuku':
            # 4 * 3
            # 29-32, 38-40
            # station (32, 40)
            self.minLat = Decimal('35.688')
            self.maxLat = Decimal('35.704')
            self.minLon = Decimal('139.69')
            self.maxLon = Decimal('139.705')
        elif region == 'ikebukuro':
            # 3 * 3
            # 21-23, 41-43
            # station (22, 42)
            self.minLat = Decimal('35.724')
            self.maxLat = Decimal('35.736')
            self.minLon = Decimal('139.705')
            self.maxLon = Decimal('139.72')
        elif region == 'shibuya':
            # 4 * 2
            # 38-41, 39-40
            # station (40, 40)
            self.minLat = Decimal('35.652')
            self.maxLat = Decimal('35.668')
            self.minLon = Decimal('139.695')
            self.maxLon = Decimal('139.705')
        elif region == 'shimbashi':
            # 3 * 2
            # 36-38, 50-51
            # station (38, 51)
            self.minLat = Decimal('35.664')
            self.maxLat = Decimal('35.676')
            self.minLon = Decimal('139.75')
            self.maxLon = Decimal('139.76')
        elif region == 'akihabara':
            # 3 * 3
            # 29-31, 53-55
            # station (30, 54)
            self.minLat = Decimal('35.692')
            self.maxLat = Decimal('35.704')
            self.minLon = Decimal('139.765')
            self.maxLon = Decimal('139.78')
        else:
            print('invalid region name...')
            assert False

        if size == '500m':
            self.dLon = Decimal('0.005')
            self.dLat = Decimal('0.004')
        elif size == '100m':
            self.dLon = Decimal('0.001')
            self.dLat = Decimal('0.0008')
        else:
            print('invalid region size...')
            assert False

        self.lonNum = int((self.maxLon - self.minLon) / self.dLon)
        self.latNum = int((self.maxLat - self.minLat) / self.dLat)
        self.size = size

        if city == 'tokyo':
            in_city = Mesh('tokyo', size)
        elif city == 'osaka':
            in_city = Mesh('osaka', size)
        else:
            print('invalid city name...')
            assert False

        self.grids = []
        self.indexs = []
        for i in range(1, self.latNum + 1):
            for j in range(1, self.lonNum + 1):
                lat = self.maxLat - i * self.dLat
                lon = self.minLon + j * self.dLon
                grid_order = in_city.inWhichGrid(Point('custom', lat, lon))
                grid_index = in_city.Index[grid_order]
                self.grids.append(grid_order)
                self.indexs.append(grid_index)


'''
Tokyo 0.32 * 0.4, 80 * 80, 35.5-35.82, 139.5-139.9
Osaka 0.32 * 0.4, 80 * 80, 34.5-34.82, 135.3-135.7
'''


class Mesh(object):
    # 0, 1, 2, 3
    # 4, 5, 6, 7
    # 8, 9, 10, 11
    # ......
    def __init__(self, city, size, lonNum, latNum):
        getcontext().prec = 8

        # (139.15, 35.266666666666666, 140.4, 36.1)
        if city == 'tokyo':
            self.minLon = Decimal('139.15')
            self.maxLon = Decimal('140.4')
            self.minLat = Decimal('35.266667')
            self.maxLat = Decimal('36.1')
        else:
            print('invalid city name...')
            assert False

        self.lonNum = lonNum
        self.latNum = latNum

        self.dLon = (self.maxLon - self.minLon) / self.lonNum
        self.dLat = (self.maxLat - self.minLat) / self.latNum
        self.size = size

        ID = 0
        self.Index = {}
        self.ReverseIndex = {}
        for i in range(self.latNum):
            for j in range(self.lonNum):
                self.Index[ID] = (i, j)
                self.ReverseIndex[(i, j)] = ID
                ID += 1

        self.meshcodes = self.toJISMesh()
        self.meshcode2id = {meshcode: i for i, meshcode in enumerate(self.meshcodes)}
        self.id2meshcode = {i: meshcode for i, meshcode in enumerate(self.meshcodes)}

    def inMesh(self, x, y):
        # x, y is index
        if x >= 0 and x < self.latNum and y >= 0 and y < self.lonNum:
            return True
        else:
            return False

    def inMeshPoint(self, point):
        # point is gps dict
        if point.lon >= self.minLon and point.lon <= self.maxLon \
                and point.lat >= self.minLat and point.lat <= self.maxLat:
            return True
        else:
            return False

    def inWhichGrid(self, point):
        # point is gps dict
        if self.inMeshPoint(point):
            x = Decimal((self.maxLat - Decimal(point.lat)) / self.dLat).quantize(Decimal('1.'), rounding=ROUND_UP) - 1
            y = Decimal((Decimal(point.lon) - self.minLon) / self.dLon).quantize(Decimal('1.'), rounding=ROUND_UP) - 1
            if x == -1:
                x = 0
            if y == -1:
                y = 0
            return self.ReverseIndex[(x, y)], x, y
        else:
            return None

    def toGPS(self, index):
        x, y = index[0], index[1]
        if self.inMesh(x, y):
            lat, lon = self.maxLat - x * self.dLat, \
                       self.minLon + y * self.dLon
            lat1, lon1 = self.maxLat - (x + 1) * self.dLat, \
                         self.minLon + (y + 1) * self.dLon
            lat, lon, lat1, lon1 = float(lat), float(lon), float(lat1), float(lon1)
            return [[lat, lon], [lat1, lon1]]
        else:
            return None

    def toJISMesh(self):
        JISMesh = []
        if self.size == '2000m':
            level = 2000
        elif self.size == '1000m':
            level = 3
        elif self.size == '500m':
            level = 4
        elif self.size == '250m':
            level = 5
        elif self.size == '100m':
            level = 6
        else:
            assert 'Invalid mesh size'

        for id in range(self.latNum * self.lonNum):
            x, y = self.Index[id]
            lat, lon = self.maxLat - x * self.dLat - Decimal('0.5') * self.dLat, \
                       self.minLon + y * self.dLon + Decimal('0.5') * self.dLon
            # lon, lat = self.minLon + x * self.dLon, \
            #                          self.minLat + y * self.dLat
            lat, lon = float(lat), float(lon)
            meshcode = ju.to_meshcode(lat, lon, level)
            JISMesh.append(meshcode)

        return JISMesh


if __name__ == '__main__':
    tokyo_mesh = Mesh('tokyo', '500m', 200, 200)
    tokyo_meshcodes = np.array(tokyo_mesh.toJISMesh())
    tokyo_meshcodes = tokyo_meshcodes.reshape(200, 200)
    tokyo_meshcodes = pd.DataFrame(tokyo_meshcodes)
    tokyo_meshcodes.to_csv('../meshcodes_40000_tokyo/meshcode3_200x200.csv', sep=',', header=None, index=False)
    print(tokyo_meshcodes.shape)
