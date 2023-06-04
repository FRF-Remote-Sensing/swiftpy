

# read radarinfo.json and get the bounding box
# Path: ataker\utils.py

import json
from shapely.geometry import Polygon
import geopandas as gpd


def readStateRadarInfo(statejsonfile):
    '''Reads the state radar info from the json file and returns a dictionary'''
    with open(statejsonfile) as data_file:
        data = json.load(data_file)
    return data

def buildbb(statejsonfile):
    '''Builds the bounding box from the radar info'''
    # read the radar info
    radarinfo = readStateRadarInfo(statejsonfile)
    # get the bounding box
    bb = radarinfo['bound_box']
    # return the bounding box
    return bbToPolygon( bb)


## bb [45.87187638752508, -123.97286414592807, 45.88626553321977, -123.9521952849472] to polygon
def bbToPolygon(bbarry):

    # get the bounding box
    xMin = bbarry[1]
    yMin = bbarry[0]
    xMax = bbarry[3]
    yMax = bbarry[2]
    # create the polygon
    polygon = Polygon([(xMin,yMin),(xMin,yMax),(xMax,yMax),(xMax,yMin), (xMin,yMin)])
    # return the polygon
    return polygon


# get intersectedpolygon from shapefile
def getIntersectedPolygon(shapefile, polygon):
    gdf = gpd.read_file(shapefile, bbox=polygon.bounds)
    # trim the bounding box to the shapefile
    gdfbb = gdf[gdf.intersects(polygon)]
    # return the trimmed bounding box

    # merge all the features into one
    gdfbb = gdfbb.unary_union

    return gdfbb


# trim bb to imput shapefile features eg. coastlines
def trimbb(bb, shapefile):
    '''Trims the bounding box to the shapefile'''
    # read the shapefile
    gdf = gpd.read_file(shapefile)
    # trim the bounding box to the shapefile
    gdfbb = gdf[gdf.intersects(bb)]
    # return the trimmed bounding box

    # merge all the features into one
    gdfbb = gdfbb.unary_union
    # return the merged feature


    return gdfbb

def getLeftoverPolygonFromIntersect(polygon1, polygon2):
    # get the intersection
    intersect = polygon1.intersection(polygon2)
    # if the intersection is not empty
    if not intersect.is_empty:
        # return the leftover
        return polygon1.difference(intersect)
    else:
        # return the original polygon
        return polygon1

# load state and trim to input shapefile
def trimmedAOI(statejsonfile, shapefile, buffer=0):
    '''Loads the state and trims to the shapefile'''
    # build the bounding box
    bb = buildbb(statejsonfile)
    # trim the bounding box to the shapefile
    intersected = getIntersectedPolygon( shapefile, bb)
    # buffer the intersected polygon if needed
    if buffer != 0:
        intersected = intersected.buffer(buffer)

    # get the leftover polygon
    leftover = getLeftoverPolygonFromIntersect(bb, intersected)

    # return the leftover polygon
    return leftover

# merge polygons in a geopandas dataframe
def mergepolygons(df, col = 'geometry'):
    '''Merges the polygons in a geopandas dataframe'''
    # get the union of all the polygons
    union = df[col].unary_union
    # get the list of polygons
    polys = list(union)
    # create a new dataframe
    dfnew = gpd.GeoDataFrame(columns = [col], crs = 4326)
    # add the polygons to the dataframe
    dfnew[col] = polys
    # return the dataframe
    return dfnew






def nbexplore_shape(shape):
    df = gpd.GeoDataFrame(columns=['geometry'], crs = 4326)
    df['geometry'] = [shape]
    return df.explore()
