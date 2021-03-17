#!/usr/bin/env python
# coding: utf-8

# Script for downloading all Swarm CT 2-Hz files from Swarm site
#
# Spencer Mark Hatch
# Birkeland Centre for Space Science
# 2021-01-18

DownloadDir = '/SPENCEdata/Research/database/Swarm/2Hz_TII_Cross-track/'
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
wantext = '.ZIP'
VERSION = '0302'

########################################
# Define which satellites we'll look for during which years 

mode = 'Annadb2'
mode = '2014'
mode = 'fulldb_5sres'
mode = 'fulldb_1sres'

VALIDMODES = ['fulldb_5sres','fulldb_1sres','Annadb','Annadb2','2014']

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_B']
# sats = ['Sat_B','Sat_C']

assert mode in VALIDMODES,"Must choose one of " + ",".join(VALIDMODES)+"!"

if mode == 'fulldb_5sres':
    hdfsuff = '_5sres'

    decimationfactor = 10           # so 5-s resolution
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

if mode == 'fulldb_1sres':
    hdfsuff = '_1sres'

    decimationfactor = 2           # so 1-s resolution
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

elif mode == 'Annadb':
    hdfsuff = '_Anna'

    decimationfactor = 1           # so full resolution (2 Hz)
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2014-01-01 00:00:00'

elif mode == 'Annadb2':
    hdfsuff = '_Anna2'

    decimationfactor = 1           # so full resolution (2 Hz)
    mlatlowlim = 45

    date0 = '2016-01-25 00:00:00'
    date1 = '2016-02-01 00:00:00'

elif mode == '2014':
    hdfsuff = '_2014'

    decimationfactor = 1           # so full resolution (2 Hz)
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2014-12-31 23:59:59'

########################################
# Imports

import cdflib
from datetime import datetime 
import numpy as np
import pandas as pd
import os
import requests
import urllib.request
from hatch_python_utils.satellites.Swarm import getCT2HzFTP,get_Swarm_combo,getCT2HzFileDateRange

from glob import glob
from hatch_python_utils.earth import coordinates as hCoord
import sys
import os
needdir = '/SPENCEdata/Research/sandbox_and_journals/journals/Swarm/'
if needdir not in sys.path:
    sys.path.append(needdir)
from swarmProcHelper import processSwarm2HzCrossTrackZip
        

########################################
# Define which satellites we'll look for during which years 

date0 = datetime.strptime(date0,"%Y-%m-%d %H:%M:%S")
date1 = datetime.strptime(date1,"%Y-%m-%d %H:%M:%S")

dates = pd.date_range(start=date0,end=date1,freq='1D')
dates = [dato.strftime("%Y%m%d") for dato in dates]

########################################
# function definitions

# def process_Swarm_CT2Hz_cdf(fn):
    
#     breakpoint()

#     getvars = ['SC_GEOCENTRIC_LAT',
#                'SC_GEOCENTRIC_LON',
#                'SC_GEOCENTRIC_R',
#                'ELE_TOTAL_ENERGY_FLUX',
#                'ELE_TOTAL_ENERGY_FLUX_STD']
    
#     cdf = cdflib.CDF(fn)
    
#     tStamps = pd.DatetimeIndex(list(map(pd.Timestamp,
#                                         cdflib.epochs.CDFepoch().to_datetime(cdf_time=cdf.varget('Epoch')))))
#     df = pd.DataFrame({var:cdf.varget(var) for var in getvars},index=tStamps)
#     # df['satid'] = satnum
    
#     return df

def store_Swarm_CT2Hz_DF(fn,df):

    store = pd.HDFStore(fn)
    for column in df.columns:
        store.append(column, df[column], format='t')

    store.close()

if VERSION == '0101':
    fullext = VERSION + '.CDF.ZIP'
else:
    fullext = VERSION + '.ZIP'

apex__geodetic2apexOpts=dict(min_time_resolution__sec=0.5,
                             max_N_months_twixt_apexRefTime_and_obs=3,
                             return_apex_d_basevecs=True)  # Need these for getting flow in Apex coordinates
########################################
# Download!
for sat in sats:

    # if sat == sats[0]:
    #     print("Already have Swarm A. Continue!")
    #     continue

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    localdir = DownloadDir+'/'.join([sat.replace('Sat_','Swarm_'),''])
    if not os.path.exists(localdir):
        print(f"Making {localdir}")
        os.makedirs(localdir)

    print(masterhdf)

    # Get times
    havemastertimes = False
    if os.path.exists(masterhdfdir+masterhdf):
        mastertimes = pd.read_hdf(masterhdfdir+masterhdf,'/mlat').copy().index
        havemastertimes = True

    # for year in years:
    curIterStr = f"{sat}-{VERSION}"

    opts_hurtigLast = dict(FP__doCorrectTimestamps=False,
                           FP__doResample=False,
                           dont_touch_data=False,
                           dontInterp__justMag=False,
                           doDebug=False,
                           overwrite_existing=False,
                           use_existing=True,
                           removeCDF=True,
                           resampleString='500ms',
                           customSaveSuff='',
                           make_pickles=False)


    fileza = glob(localdir+'*.ZIP')
    fileza.sort()

    timeranges = [getCT2HzFileDateRange(os.path.basename(fila)) for fila in fileza]

    for fila,tidrange in zip(fileza,timeranges):
        dirrie = os.path.dirname(fila)+'/'

        if (tidrange[0] < date0) or (tidrange[0] > date1):
            continue

        if havemastertimes:

            if np.sum((mastertimes >= tidrange[0]) & (mastertimes <= tidrange[1])):
                print(f"Already have {os.path.basename(fila)}! Continue ...")
                continue

        df = processSwarm2HzCrossTrackZip(dirrie, os.path.basename(fila), localdir,
                                          doResample=False,
                                          resampleString="62500000ns",
                                          skipEphem=True,
                                          quiet=False,
                                          removeCDF=True,
                                          rmCDF_noPrompt=True,
                                          dont_touch_data=False,
                                          include_explicit_calibration_flags=False)

        # if len(df) != 0:
        if df is not None:
            df.sort_index(inplace=True)
            
            # Decimation 
            if decimationfactor > 1:
                df = df.iloc[::decimationfactor]


            # print("Adding Apex coords")
            
            gdlat, gdalt_km = hCoord.geoclatR2geodlatheight(
                df["Latitude"].values, df["Radius"].values/1000.)

            apexDict2 = hCoord.geodetic2apex(gdlat, df["Longitude"].values,
                                             gdalt_km,
                                             df.index.to_pydatetime(),
                                             **apex__geodetic2apexOpts)

            df = df.assign(**apexDict2)

            df = df[np.abs(df['mlat']) >= mlatlowlim]

            # Add this DataFrame to master .h5 file
            store_Swarm_CT2Hz_DF(masterhdfdir+masterhdf,df)

    #     # Delete .cdf file
    #     # os.remove(localdir+fil)

    # print("")

    # break


# TEST READING HDF
# with pd.HDFStore(localdir+masterhdf, mode = 'r') as store:
#     indata = store[satellite + '/raw_data']

# print("Reading HDF file ...")
# with pd.HDFStore(localdir+masterhdf, mode='r') as hdf:
#     # This prints a list of all group names:
#     keys = hdf.keys()

# indata = pd.DataFrame({key.replace('/', ''): pd.read_hdf(localdir+masterhdf, key=key).values for key in keys},
#                       index=pd.read_hdf(localdir+masterhdf, key=keys[0]).index)

