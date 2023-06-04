

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    import math

    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def latlon2frf(lon,lat):

    import pyproj
    import numpy as np
    # EPSG = 3358  # taken from spatialreference.org/ref/epsg/3358
    # NC stateplane NAD83
    spNC = pyproj.Proj("EPSG:3358")
    spE, spN = spNC(lon ,lat)

    r2d = 180.0 / np.pi;
    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d

    # to FRF coords
    spLengE = spE - Eom
    spLengN = spN - Nom
    R = np.sqrt(spLengE ** 2 + spLengN ** 2)
    Ang1 = np.arctan2(spLengE, spLengN)
    Ang2 = Ang1 + spAngle
    # to FRF
    X = R * np.sin(Ang2)
    Y = R * np.cos(Ang2)
    # to Lat Lon

    ans = {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN, 'X': X, 'Y': Y}
    return ans


def LatLon2ncsp(lon, lat):
    import pyproj
    """This function uses pyproj to convert longitude and latitude to stateplane
        
        test points taken from conversions made in USACE SMS modeling system
        
            nc stateplane  meters NAD83
            spE1 = 901926.2 m
            spN1 = 273871.0 m
            Lon1 = -75.75004989
            Lat1 =  36.17560399
        
            spE2 = 9025563.9 m
            spN2 = 276229.5 m
            lon2 = -75.47218285
            lat2 =  36.19666112

        Args:
        lon: geographic longitude (NAD83)  decimal degrees
        lat: geographic longitude (NAD83)  decimal degrees

        Returns:
        output dictionary with original coords and output of NC stateplane FIPS 3200
            'lat': latitude

            'lon': longitude

            'StateplaneE': NC stateplane

            'StateplaneN': NC stateplane

    """
    #EPSG = 3358  # taken from spatialreference.org/ref/epsg/3358
    # NC stateplane NAD83
    spNC = pyproj.Proj("EPSG:3358")
    spE, spN = spNC(lon,lat)
    ans = {'lon': lon, 'lat': lat, 'StateplaneE': spE, 'StateplaneN': spN}
    return ans



def ncsp2FRF(p1, p2):
    import numpy as np
    """this function converts nc StatePlane (3200 fips) to FRF coordinates
    based on kent Hathaways Code
    #
    #  15 Dec 2014
    #  Kent Hathaway.
    #  Translated from Matlab to python 2015-11-30 - Spicer Bak
    #
    #  Uses new fit (angles and scales) Bill Birkemeier determined in Nov 2014
    #
    #  This version will determine the input based on values, outputs FRF, lat/lon,
    #  and state plane coordinates.  Uses NAD83-2011.
    #
    #  IO:
    #  p1 = FRF X (m), or Longitude (deg + or -), or state plane Easting (m)
    #  p2 = FRF Y (m), or Latitude (deg), or state plane Northing (m)
    #
    #  X = FRF cross-shore (m)
    #  Y = FRF longshore (m)
    #  ALat = latitude (decimal degrees)
    #  ALon = longitude (decimal degrees, positive, or W)
    #  spN = state plane northing (m)
    #  spE = state plane easting (m)
    
    NAD83-86	2014
    Origin Latitude       36.1775975
    Origin Longitude      75.7496860
    m/degLat              110963.357
    m/degLon               89953.364
    GridAngle (deg)          18.1465
    Angle FRF to Lat/Lon     71.8535
    Angle FRF to State Grid  69.9747
    FRF Origin Northing  274093.1562
    Easting              901951.6805
    
    #  Debugging values
    p1=566.93;  p2=515.11;  % south rail at 1860
    ALat = 36.1836000
    ALon = 75.7454804
    p2= 36.18359977;
    p1=-75.74548109;
    SP:  p1 = 902307.92;
    p2 = 274771.22;

    Args:
      spE: North carolina state plane coordinate system - Easting
      spN: North carolina state plane coordinate system - Northing
      p1: first point
      p2: second point

    Returns:
      dictionary
       'xFRF': cross shore location in FRF coordinates

       'yFRF': alongshore location in FRF coodrindate system

       'StateplaneE': north carolina state plane coordinate system - easting

       'StateplaneN': north carolina state plane coordinate system - northing

    """
    r2d = 180.0 / np.pi;
    Eom = 901951.6805;  # % E Origin State Plane
    Nom = 274093.1562;  # % N Origin State Plane
    spAngle = (90 - 69.974707831) / r2d

    spE = p1
    spN = p2  # designating stateplane vars


    # to FRF coords
    spLengE = p1 - Eom
    spLengN = p2 - Nom
    R = np.sqrt(spLengE ** 2 + spLengN ** 2)
    Ang1 = np.arctan2(spLengE, spLengN)
    Ang2 = Ang1 + spAngle
    # to FRF
    X = R * np.sin(Ang2)
    Y = R * np.cos(Ang2)
    # to Lat Lon
    ans = {'xFRF': X,
           'yFRF': Y,
           'StateplaneE': spE,
           'StateplaneN': spN}
    return ans

# ncSP = LatLon2ncsp(-75.75004989,36.17560399)
#
# print(ncSP)
#
# frfCoords = ncsp2FRF(ncSP['StateplaneE'],ncSP['StateplaneN'])
#
# print(frfCoords)


def roundgrid(x,y,z,xgrid,ygrid):
    import numpy as np
    # x = newX
    # y = newY
    # z = np.asarray(data[0].value)
    #
    ymax = np.max(ygrid.flatten())
    xmax = np.max(xgrid.flatten())
    ymin = np.min(ygrid.flatten())
    xmin = np.min(ygrid.flatten())
    # calculate limits of the grid
    if np.diff(ygrid[:,0])[0] == 0:
        dy = np.diff(ygrid[0:1])[0][0]
        dx = np.diff(xgrid[:,0])[0]
    else:
        dx = np.diff(xgrid[0:1])[0][0]
        dy = np.diff(ygrid[:,0])[0]
    # convert to 2D index space
    X = np.round(x/dx - (xmin/dx-1))
    Y = np.round(y/dy - (ymin/dy-1))
    # find point out of the grid
    m,n = np.shape(xgrid)
    outsideInds = np.where((X<1) | (X>(n-1)) | (Y<1) | (Y>(m-1)))
    X = np.delete(X,outsideInds)
    Y = np.delete(Y,outsideInds)
    Z = np.delete(z,outsideInds)

    def ind2sub(array_shape, ind):
        ind[ind < 0] = -1
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        rows = np.floor((ind.astype('int') / array_shape[1]))
        cols = ind % array_shape[1]
        return (rows.astype(int), cols)

    def sub2ind(array_shape, rows, cols):
        ind = rows*array_shape[1] + cols
        ind[ind < 0] = -1
        ind[ind >= array_shape[0]*array_shape[1]] = -1
        return ind
    # convert to a 1-D index
    ind = sub2ind([m,n],Y,X)
    # sort index and zs by index number
    sortedIndex = np.argsort(ind,axis=0)
    sortedZs = Z[sortedIndex]
    sortedInd = ind[sortedIndex].astype(int)
    # calculate the difference in index number... diff of 0 means its the same index
    di = np.hstack((1,np.diff(sortedInd)))
    # preallocate
    zgrid = np.zeros((m,n))
    numpts = np.zeros((m,n))


    while np.size(Z) > 0:
        dinot0 = np.where(di!=0) # if its not 0, process it
        idi = sortedInd[dinot0] # index number of those values
        id2 = ind2sub([m,n],idi) # subindices of those matrix locations
        # each index is going to have one more point included in the mean
        numpts[id2] = numpts[id2]+1
        # calculate a running mean?
        zgrid[id2] = Z[dinot0] * (1/numpts[id2]) + zgrid[id2] * ((numpts[id2]-1)/numpts[id2])
        Z = np.delete(Z,dinot0)
        sortedInd = np.delete(sortedInd,dinot0)
        di = np.hstack((1,np.diff(sortedInd)))
        # print(np.shape(Z),np.shape(di),np.shape(sortedInd),np.max(idi))

    return zgrid