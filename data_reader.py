# data reader function for ataker pkl data
import bz2
from ctypes import Structure, c_byte, c_uint16, c_uint32, c_uint8
import datetime
import pickle
import pandas as pd
import geopandas as gpd
import h3
import shapely.geometry as geometry
import numpy as np
import io

import sys

from ataker.boundingbox import get_bounding_box
import ataker


# sys.modules['radarrotation'] = ataker
# sys.modules['radarcontrol'] = ataker


class SimpleRot:
    def __init__(self, *args, **kwargs):
        self.heading = 0.0
        self.lat = 0.0
        self.lng = 0.0
        self.range = 0.0
        self.lines = []
        self.timestamp = datetime.datetime.now()


class RadarLine(Structure):
    _pack_ = 1
    _fields_ = [
        ("HeaderLen", c_byte),
        ("Status", c_byte),
        ("ScanNumber", c_uint16),
        ("u00", c_uint16),
        ("largerange", c_uint16),
        ("Angle", c_uint16),
        ("Heading", c_uint16),
        ("smallrange", c_uint16),
        ("rotation", c_uint8 * 2),
        ("u02", c_uint32),
        ("u03", c_uint32),
        ("DataBytes", c_uint8 * 512),
    ]


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):

        ## log each type out
        print(module, name)
        if module == 'radarrotation' and name == 'SimpleRot':
            return SimpleRot

        if module == 'radarcontrol' and name == 'RadarLine':
            return RadarLine

        return super().find_class(module, name)


def h3toshapelypolygon(h):
    try:
        h3poly = h3.cell_to_boundary(h, geo_json=True)
        return geometry.Polygon([[p[0], p[1]] for p in h3poly])
    except:
        return None


def convert(pklfile):
    data = CustomUnpickler(open(pklfile, 'rb')).load()
    for rot in data:
        print(rot.loc.lat, rot.loc.lng, rot.range)
        # if(count > 15):
        #     break
        # bb = get_bounding_box(rot.loc.lat, rot.loc.lng, rot.range)

        # rot.process()

        # frame = rot.map_matrix.astype('uint8')
        # alpha = np.sum(frame, axis=-1) > 0
        # alpha = np.uint8(alpha * 255)
        # frame = np.dstack((frame, alpha))


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def read_data(ataker_df, res=15, frames=360, trimpoly=None):
    """Read data from data file."""
    print(f"Reading data from {ataker_df}")

    # read pickle file to byte array
    with open(ataker_df, 'rb') as f:
        data = f.read()
        data = CustomUnpickler(io.BytesIO(data)).load()

    # for each data frame in the array render to geopandas dataframe with h3
    outdata = []

    for i in range(len(data)):
        if i > frames:
            break

        # revmove rows with value = 0
        data[i] = data[i][data[i].value != 0]

        # remove dups on lat lng
        data[i] = data[i].drop_duplicates(subset=['x', 'y'])

        # if trim poly remove any data outside of it by lat lng
        if trimpoly is not None:
            # make a geodataframe from data[i] by lat lng
            data[i] = gpd.GeoDataFrame(data[i], geometry=gpd.points_from_xy(data[i].x, data[i].y), crs=4326)
            trimdf = gpd.GeoDataFrame({"geometry": [trimpoly]}, geometry='geometry', crs=4326)
            data[i] = gpd.sjoin(data[i], trimdf, how="inner", predicate='intersects')

            # h3 @ resolution 14
        data[i]["h3_render"] = [h3.latlng_to_cell(xy[1], xy[0], res) for xy in zip(data[i].x, data[i].y)]

        # remove duplicates on h3_render
        data[i]["h3_render"] = data[i]["h3_render"].drop_duplicates()

        # data[i]["h3_render"] = [h3.latlng_to_cell(xy[1], xy[0], res) for xy in zip(data[i].x, data[i].y)]
        # data[i]["h3_render"] = data[i]["h3_render"].drop_duplicates()

        data[i]["geometry"] = [h3toshapelypolygon(h) for h in data[i]["h3_render"]]

        # remove null geometries
        data[i] = data[i][data[i].geometry.notnull()]

        data[i] = gpd.GeoDataFrame(data[i], geometry='geometry', crs=4326)

        # #if trim poly remove any data outside of it
        # if trimpoly is not None:
        #     trimdf = gpd.GeoDataFrame({"geometry" :[ trimpoly]}, geometry='geometry', crs = 4326 )
        #     data[i] = gpd.sjoin(data[i], trimdf, how="inner", predicate='intersects')

        outdata.append(data[i])

    return outdata
