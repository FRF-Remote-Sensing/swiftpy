
import datetime
import pickle
import io
import numpy as np
import numpy as np
from functions import roundgrid
from functions import rotate
from functions import latlon2frf
import matplotlib.pyplot as plt
import geopandas as gpd

rotateAboutX = 65
rotateAboutY = 580
rotateAngle = -38

frames = 599
trimpoly = None
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230524-1530_frf_dunetoe_r800_100rots/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230525-1345_frf_dunetoe_r800_600rots_gainHI/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230525-1330_frf_dunetoe_r800_360rots_gainNOM/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20250525-1400_frf_dunetoe_r800_600rots_gainLO/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230601-1240_frf_dunetoe_r800_600rots_gain60_FOG/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230602-1320_frf_dunetoe_r800_600rots_gain75_FOG/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230601-1315_frf_dunetoe_r400_600rots_gain75_FOG/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230531-1845_frf_dunetoe_r800_600rots_gain65/record.pkl' # raw data file from radar
# ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230526-1830_frf_dunetoe_r800_600rots_gain60ff/record.pkl' # raw data file from radar

ataker_df = '/Users/dylananderson/Documents/projects/ataker/20230526-1440_frf_dunetoe_r400_600rots_gain60ff/record.pkl' # raw data file from radar

print(f"Reading data from {ataker_df}")

# read pickle file to byte array
with open(ataker_df, 'rb') as f:
    data = f.read()
    data = pickle.Unpickler(io.BytesIO(data)).load()


xInterp = np.arange(0, 600, 2)
yInterp = np.arange(0, 1000, 2)
xgrid, ygrid = np.meshgrid(xInterp, yInterp)
m,n = np.shape(xgrid)
rectData = np.zeros((m,n,frames))
smoothI = np.zeros((m,n,frames))
smoothI2 = np.zeros((m,n,frames))

for qq in range(frames):
    print('processing rotation #{}'.format(qq))
    frf = latlon2frf(data[qq].x,data[qq].y)
    newPoint = [rotate((rotateAboutX,rotateAboutY),(frf['X'][hh],frf['Y'][hh]),rotateAngle*np.pi/180) for hh in range(len(data[qq]))]
    newX = np.asarray([x[0] for x in newPoint])
    newY = np.asarray([x[1] for x in newPoint])
    zgrid = roundgrid(x=newX, y=newY, z=np.asarray(data[qq].value), xgrid=xgrid, ygrid=ygrid)
    rectData[:,:,qq] = zgrid

    import scipy.ndimage
    import numpy as np

    indx = np.where(np.isnan(zgrid));
    # smoothZ = scipy.ndimage.correlate(zgrid, np.ones((2,5)), mode='constant')
    # smoothZ = scipy.ndimage.correlate(zgrid, np.ones((5,2)), mode='reflect')
    # smoothZ = scipy.ndimage.gaussian_filter(zgrid, (5,2))
    smoothZ = scipy.ndimage.gaussian_filter(zgrid, (4,1))

    smoothI[:,:, qq] = smoothZ#filt2d(zgrid, 'avgnans', 5, 2);
    # smoothI2[:,:, qq] = smoothZ2#filt2d(zgrid, 'avgnans', 5, 2);

    # h = ones(r, c)
    #
    # k = imfilter(double(~indx), h);
    # Iin(indx) = 0;
    # ISum = imfilter(Iin, h);
    # Iout = ISum. / k;
    # indx = [];
    # Iout(indx) = nan;



    # plt.figure()
    # plt.scatter(data[0].x, data[0].y, s=10,c=data[0].value)
    # # plt.scatter(frf['X'], frf['Y'], s=10,c=data[0].value)
    # plt.show()
    # plt.figure()
    # plt.scatter(newX, newY, s=10,c=data[0].value)
    # plt.show()

    rot = qq
    f1 = plt.figure(figsize=(13, 5))
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    s1 = ax1.scatter(data[qq].x, data[qq].y, s=10, c=data[qq].value, vmin=0, vmax=15)
    ax1.axis('equal')
    ax1.set_xlabel('raw Lon')
    ax1.set_ylabel('raw Lat')
    cb1 = plt.colorbar(s1)
    cb1.set_label('Raw Returns')
    ax2 = plt.subplot2grid((1, 3), (0, 1))
    p1 = ax2.pcolor(xgrid, ygrid, rectData[:, :, rot], vmin=0, vmax=15)
    cb2 = plt.colorbar(p1)
    cb2.set_label('Gridded Average')
    ax2.axis('equal')
    ax2.set_xlabel('xFRF (m)')
    ax2.set_ylabel('yFRF (m)')
    ax3 = plt.subplot2grid((1, 3), (0, 2))
    p2 = ax3.pcolor(xgrid, ygrid, smoothI[:, :, rot], vmin=0, vmax=10)
    cb3 = plt.colorbar(p2)
    cb3.set_label('Gaussian Filter')
    ax3.axis('equal')
    ax3.set_xlabel('xFRF (m)')
    ax3.set_ylabel('yFRF (m)')
    plt.tight_layout()

    #
    if qq < 10:
        plt.savefig('/Users/dylananderson/Documents/projects/ataker/20230526-1440_frf_dunetoe_r400_600rots_gain60ff/videoFrames/frame_00{}'.format(qq))
    elif qq < 100:
        plt.savefig('/Users/dylananderson/Documents/projects/ataker/20230526-1440_frf_dunetoe_r400_600rots_gain60ff/videoFrames/frame_0{}'.format(qq))
    else:
        plt.savefig('/Users/dylananderson/Documents/projects/ataker/20230526-1440_frf_dunetoe_r400_600rots_gain60ff/videoFrames/frame_{}'.format(qq))
    plt.clf()
    plt.close(f1)
    del frf
    del zgrid
    del newPoint
    del newX
    del newY



iGridMean = np.nanmean(rectData,axis=2)
iGridMean2 = np.nanmean(smoothI,axis=2)
f2 = plt.figure(figsize=(10,5))
ax1 = plt.subplot2grid((1, 2), (0, 0))
p10 = ax1.pcolor(xgrid, ygrid, iGridMean, vmin=0, vmax=10)
ax1.axis('equal')
ax1.set_xlabel('xFRF (m)')
ax1.set_ylabel('yFRF (m)')
cb1 = plt.colorbar(p10)
cb1.set_label('Mean of Gridded Product')
ax2 = plt.subplot2grid((1, 2), (0, 1))
p11 = ax2.pcolor(xgrid, ygrid, iGridMean2, vmin=0, vmax=10)
ax2.axis('equal')
ax2.set_xlabel('xFRF (m)')
ax2.set_ylabel('yFRF (m)')
cb2 = plt.colorbar(p11)
cb2.set_label('Mean of Gaussian Smoothed')



[F1,F2] = np.gradient(smoothI[:,:, 0], edge_order=2)






rot = 5
f1 = plt.figure(figsize=(13, 5))
ax1 = plt.subplot2grid((1, 3), (0, 0))
s1 = ax1.scatter(data[qq].x, data[qq].y, s=10, c=data[qq].value, vmin=0, vmax=15)
ax1.axis('equal')
ax1.set_xlabel('raw Lon')
ax1.set_ylabel('raw Lat')
cb1 = plt.colorbar(s1)
cb1.set_label('Raw Returns')
ax2 = plt.subplot2grid((1, 3), (0, 1))
p1 = ax2.pcolor(xgrid, ygrid, rectData[:,:,rot], vmin=0, vmax=15)
cb2 = plt.colorbar(p1)
cb2.set_label('Gridded Average')
ax2.axis('equal')
ax2.set_xlabel('xFRF (m)')
ax2.set_ylabel('yFRF (m)')
ax3 = plt.subplot2grid((1, 3), (0, 2))
p2 = ax3.pcolor(xgrid, ygrid, smoothI[:,:,rot], vmin=0, vmax=10)
cb3 = plt.colorbar(p2)
cb3.set_label('Gaussian Filter')
ax3.axis('equal')
ax3.set_xlabel('xFRF (m)')
ax3.set_ylabel('yFRF (m)')
plt.tight_layout()
# plt.show()

# f1 = plt.figure(figsize=(10,5))
# ax1 = plt.subplot2grid((1,2),(0,0))
# s1 = ax1.scatter(data[qq].x, data[qq].y, s=10,c=data[qq].value,vmin=0,vmax=15)
# ax1.axis('equal')
# ax1.set_xlabel('raw Lon')
# ax1.set_ylabel('raw Lat')
# cb1 = plt.colorbar(s1)
# cb1.set_label('Raw Returns')
# ax2 = plt.subplot2grid((1,2),(0,1))
# p1 = ax2.pcolor(xgrid,ygrid,zgrid,vmin=0,vmax=15)
# cb2 = plt.colorbar(p1)
# cb2.set_label('Gridded Average')
# ax2.axis('equal')
# ax2.set_xlabel('xFRF (m)')
# ax2.set_ylabel('yFRF (m)')
# plt.tight_layout()
# #plt.show()

#
# # have the bytes... what next?
#
# # for each data frame in the array render to geopandas dataframe with h3
# outdata = []
#
# for i in range(len(data)):
#     if i > frames:
#         break
#
#     # revmove rows with value = 0
#     data[i] = data[i][data[i].value != 0]
#
#     # remove dups on lat lng
#     data[i] = data[i].drop_duplicates(subset=['x', 'y'])
#
#     # if trim poly remove any data outside of it by lat lng
#     if trimpoly is not None:
#         # make a geodataframe from data[i] by lat lng
#         data[i] = gpd.GeoDataFrame(data[i], geometry=gpd.points_from_xy(data[i].x, data[i].y), crs=4326)
#         trimdf = gpd.GeoDataFrame({"geometry": [trimpoly]}, geometry='geometry', crs=4326)
#         data[i] = gpd.sjoin(data[i], trimdf, how="inner", predicate='intersects')

import os
import cv2
satdir = '/Users/dylananderson/Documents/projects/ataker/20230526-1440_frf_dunetoe_r400_600rots_gain60ff/videoFrames/'
files = os.listdir(satdir)
files.sort()
files_path = [os.path.join(satdir,x) for x in os.listdir(satdir)]
files_path.sort()
frame = cv2.imread(files_path[0])
height, width, layers = frame.shape
# forcc = cv2.VideoWriter_fourcc(*'XVID')
forcc = cv2.VideoWriter_fourcc(*'MJPG')

video = cv2.VideoWriter('radarReturns_20230526_1440_gain60_r400.avi', forcc, 10, (width, height))
for image in files_path:
    video.write(cv2.imread(image))
cv2.destroyAllWindows()
video.release()

