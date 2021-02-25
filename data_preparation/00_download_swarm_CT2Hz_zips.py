#!/usr/bin/env python
# coding: utf-8

# Script for downloading all Swarm CT 2-Hz files from Swarm site
#
# Spencer Mark Hatch
# Birkeland Centre for Space Science
# 2021-01-18

DownloadDir = '/SPENCEdata/Research/database/Swarm/2Hz_TII_Cross-track/'
wantext = '.ZIP'
VERSION = '0302'

########################################
# Imports

from bs4 import BeautifulSoup
import cdflib
from datetime import datetime 
import numpy as np
import pandas as pd
import os
import requests
import urllib.request
from hatch_python_utils.satellites.Swarm import getCT2HzFTP,get_Swarm_combo,getCT2HzFileDateRange

########################################
# Define which satellites we'll look for during which years 

sats = ['Sat_A','Sat_B','Sat_C']
y0,y1 = 2013,2021
years = [str(val) for val in np.arange(y0,y1)]

date0 = datetime(y0,1,1,0)
date0 = datetime(2013,12,1,0)
date1 = datetime(y1,1,1,0)
dates = pd.date_range(start=date0,end=date1,freq='1D')
dates = [dato.strftime("%Y%m%d") for dato in dates]

########################################
# function definitions

def get_remote_swarm_ct2hz_dir(*args):
    """
    remoteDir = get_remote_swarm_ct2hz_dir(sat,year)
    """
    BaseAddr = 'swarm-diss.eo.esa.int/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/'
    sat = args[0]
    version = args[1]
    print(BaseAddr)
    assert version in ['0101','0201','0301','0302'], "PWHAT?"
        
    if version == '0302':
        paf = 'New_baseline'
    else:
        paf = 'Old_baseline'
        assert 2<1,"Can't do old calibrations! FTP hasn't worked for some time now (20210224), and current implementation only downloads new files ..."
    # year = args[1]
    return BaseAddr + '/'.join([paf,sat,''])


def get_url_paths(url, ext='', params={}, debug=False):
    """
    Does nothing more than find out what filenames exist at a particular url
    """
    if debug:
        print("DEBUG: url = {:s}".format(url))
    response = requests.get(url, params=params, timeout=10)
    if response.ok:
        response_text = response.text
    else:
        return response.raise_for_status()

    soup = BeautifulSoup(response_text, 'html.parser')
    parent = [node.get('href') for node in soup.find_all('a') if node.get('href') is not None]
    if ext != '':
        parent = [thing for thing in parent if thing.endswith(ext)]


    return parent


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


def get_files_we_dont_have(fn,checkfiles):

    # Open master hdf
    # fn = localdir+masterhdf

    # Check if HDF exists!
    if not os.path.exists(fn):
        print(f"{fn} does not exist!")
        return checkfiles

    with pd.HDFStore(fn, mode='r') as hdf:
        keys = hdf.keys()


    indices = pd.DataFrame({key.replace('/', ''): pd.read_hdf(fn, key=key).values for key in [keys[0]]},
                           index=pd.read_hdf(fn, key=keys[0]).index).index

    # Dates in this HDF file
    # havedates = np.unique(indices.date)

    # Loop over provided filenames to see if they are already here
    wantfiles = []
    wantfiledict = dict()
    for fil in checkfiles:

        dtrange = getCT2HzFileDateRange(fil)
        havefile = ((indices >= dtrange[0]) & (indices <= dtrange[1])).sum() > 0
        # datestr = fil.split('_')[-2]
        # dt = datetime.strptime(datestr,"%Y%m%d").date()

        wantfiledict[fil] = havefile
        if not havefile:
            wantfiles.append(fil)

    print(f"Already had {len(checkfiles)-len(wantfiles)} of {len(checkfiles)} total files")

    # return wantfiles
    return wantfiles,wantfiledict

if VERSION == '0101':
    fullext = VERSION + '.CDF.ZIP'
else:
    fullext = VERSION + '.ZIP'

########################################
# Download!
for sat in sats:

    print(sat)

    # masterhdf = sat+f'_ct2hz_v{VERSION}.h5'

    localdir = DownloadDir+'/'.join([sat.replace('Sat_','Swarm_'),''])
    if not os.path.exists(localdir):
        print(f"Making {localdir}")
        os.makedirs(localdir)

    # print(masterhdf)

    # for year in years:
    curIterStr = f"{sat}-{VERSION}"

    # opts_hurtigLast = dict(FP__doCorrectTimestamps=False,
    #                        FP__doResample=False,
    #                        dont_touch_data=False,
    #                        dontInterp__justMag=False,
    #                        doDebug=False,
    #                        overwrite_existing=False,
    #                        use_existing=True,
    #                        removeCDF=True,
    #                        resampleString='500ms',
    #                        customSaveSuff='',
    #                        make_pickles=False)

    # dads = get_Swarm_combo(dates,
    #                        sat=sat.replace('Sat_',''),
    #                        get_dtypes=['CT2HZ'],
    #                        dtype__add_Apex=['CT2HZ'],
    #                        apex__geodetic2apexOpts=dict(min_time_resolution__sec=0.5,
    #                                                     max_N_months_twixt_apexRefTime_and_obs=3),
    #                        localSaveDir='/media/spencerh/data/Swarm/',
    #                        # check_for_existing_funcDict=dict(),
    #                        removeDownloadedZipAfterProcessing=False,
    #                        useKallesHardDrive=False,
    #                        only_download_zips=True,
    #                        only_list=False,
    #                        only_remove_zips=False,
    #                        opts_hurtigLast=opts_hurtigLast)

    ########################################
    ########################################
    # SINCE FTP DOESN'T WORK, TRY THIS

    # Read in manually snatched list of files

    print("20210224 Even my new wget-based method doesn't work. The FTP site seems to hate me now, and I don't know why.")

    import wget
    from glob import glob
    filelistdir = '/SPENCEdata/Research/SHEIC/filelists/'

    swarmFTPAddr = "swarm-diss.eo.esa.int"
    subfolder = 'New_baseline' 

    sattie = sat.replace('Sat_','')
    flist = glob(filelistdir+f'{sattie}*.txt')

    assert len(flist) == 1

    with open(flist[0]) as f:
        flist = f.read().splitlines()

    todownload = []
    for f in flist:
        # print(f,end=' ')
        if os.path.exists(localdir+f):

            if os.path.getsize(localdir+f) < 900: # in bytes
                print(f"{f} is teensy-weensy! Removing and downloading again ...")
                os.remove(localdir+f)    
            else:
                # print("Skipping!")
                continue

        # print(f"Adding {f}")
        todownload.append(f)
    
    if len(todownload) == 0:
        print(f"Already have all files for {sat}. Continuing ...")
        continue

    cont = False
    while not cont:
        response = input(f"Going to download {len(todownload)} files. Sound OK? [y/n/(s)how me]")
        if len(response) == 0:
            continue
        response = response.lower()[0]
        cont = response in ['y','n']
        
        if not cont:
            if response == 's':
                [print(f) for f in todownload]
            else:
                print("Invalid. Say 'yes' or 'no'")

    if response == 'n':
        print("OK, skipping ...")
        continue

    print("Downloading!")

    subDir = f'/Advanced/Plasma_Data/2Hz_TII_Cross-track_Dataset/{subfolder:s}/Sat_{sattie:s}/'

    for ftpFile in todownload:
        print(f"Downloading {ftpFile}")
        wget.download('ftp://'+swarmFTPAddr+subDir+ftpFile,localdir+ftpFile)
        # print(f"Would run this: wget.download('{'ftp://'+swarmFTPAddr+subDir+ftpFile,localdir+ftpFile}')")
    

    # break
    # TMPVREAK

    # dads = dads[0]             # stupid nested list ...
    # dads = [os.path.basename(dad) for dad in dads]
    # # if len(dads) == 0:
    # #     continue
    # # elif len(dads) == 1:
    # #     if len(dads[0]) == 0:
    # #         continue
            
    # #     getCT2HzFTP(sat=sat.replace('Sat_',''),
    # #                 dates=[dato],
    # #                 localSaveDir=DownloadDir,
    # #                 calversion=VERSION,
    # #                 only_list=False,
    # #                 check_for_existing_func=None,
    # #                 append_dir=False)

    # # wantfiles = get_files_we_dont_have(localdir+masterhdf,dads)
    # wantfiles,wantfiledict = get_files_we_dont_have(localdir+masterhdf,dads)

    # dts = [getCT2HzFileDateRange(fName) for fName in dads]

    # breakpoint()

    # datedict = dict()
    # for dt in dts:
    #     datestr = dt[0].strftime("%Y%m%d")
    #     if datestr in datedict:
    #         datedict[datestr] += 1
    #     else:
    #         datedict[datestr] = 1

    # # breakpoint()

    # # wantfiles = [fil for fil in files if not os.path.exists(localdir+fil)]
    # # if len(wantfiles) == 0:
    # #     print("Already have all files for {curIterStr}")
    # #     continue

    # print(f"Downloading {len(wantfiles)} files for {curIterStr}")
    # # for fil in wantfiles:
    # for idate,dato in enumerate(datedict.keys()):
    #     print(dato,end=',',flush=True)
    #     tmpdfs = get_Swarm_combo([dato],
    #                            sat=sat.replace('Sat_',''),
    #                            get_dtypes=['CT2HZ'],
    #                            dtype__add_Apex=['CT2HZ'],
    #                            apex__geodetic2apexOpts=dict(min_time_resolution__sec=0.5,
    #                                                         max_N_months_twixt_apexRefTime_and_obs=3),
    #                            localSaveDir='/media/spencerh/data/Swarm/',
    #                            # check_for_existing_funcDict=dict(),
    #                            removeDownloadedZipAfterProcessing=True,
    #                            useKallesHardDrive=False,
    #                            only_download_zips=False,
    #                            only_list=False,
    #                            only_remove_zips=False,
    #                            opts_hurtigLast=opts_hurtigLast)
        
    #     # if datedict[dato] > 1:
    #     #     print("GT 1!")
    #     #     breakpoint()

    #     if isinstance(tmpdfs,list) and (len(tmpdfs) > 1):
    #         tmpdfs = pd.concat(tmpdfs)
    #     elif isinstance(tmpdfs,list):
    #         tmpdfs = tmpdfs[0]

    #     # if idate < 2:
    #     #     breakpoint()

    #     # urllib.request.urlretrieve(remoteDir+fil, localdir+fil)

    #     # Open CDF file and get desired data
    #     # tmpdf = process_Swarm_CT2Hz_cdf(localdir+fil)
        
    #     # Add this DataFrame to master .h5 file
    #     store_Swarm_CT2Hz_DF(localdir+masterhdf,tmpdfs)

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

