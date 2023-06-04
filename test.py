#%%
print('ATAK-ER Development')
print('----- steps -----')
print('1) Need to add this working directory')
import sys
import os
import numpy as np

print(os.getcwd())
sys.path.append(os.getcwd())

print('2) Need to load whatever ATAK-ER saves')

print('3) Need to rectify data to FRF coordinates')
print('   - don''t know what coord system atak spits out')
print('   - but assuming lat/lon needs to become local FRF')

from functions import LatLon2ncsp, ncsp2FRF
# ncSP = LatLon2ncsp(-75.75004989,36.17560399)
# frfCoords = ncsp2FRF(ncSP['StateplaneE'],ncSP['StateplaneN'])

# #%%
# import matplotlib.pyplot as plt
# import numpy as np
# plt.figure()
# plt.plot(np.arange(1,10),np.arange(1,10))




# %%
import json
# Opening JSON file
f = open('/Users/dylananderson/Documents/projects/ataker/20230509_frf_1100/project.json')
# returns JSON object as a dictionary
data = json.load(f)
# Closing file
f.close()


# %%
import pickle
import pandas as pd
from ataker import data_reader, utils, frame_compare
dataFile = '/Users/dylananderson/Documents/projects/ataker/20230524-1530_frf_dunetoe_r800_100rots/record.pkl' # raw data file from radar
res = 4 # processing resolution... affects the speed of the processing
frames = 10 # subset of frames from 0 to process
#trimpoly = aoi # optional parameter
live_data = data_reader.read_data(dataFile,res,frames)


# with open(r"/Users/dylananderson/Documents/projects/ataker/20230509_frf_1100/record.pkl", "rb") as input_file:
#    data = pickle.load(input_file)


# %%
import pickle
# dataFile = '/Users/dylananderson/Documents/projects/ataker/20230523-1930_frf_dunetoe/dfrecord.pkl' # raw data file from radar
dataFile = '/Users/dylananderson/Documents/projects/ataker/20230524-1530_frf_dunetoe_r800_100rots/dfrecord.pkl' # raw data file from radar

with open(dataFile, "rb") as input_file:
    df = pickle.load(input_file)

frame1 = df[0]
temp = frame1.return_values.values
# %%
