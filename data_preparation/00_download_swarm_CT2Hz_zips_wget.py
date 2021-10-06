#!/usr/bin/env python
# coding: utf-8

# wget-based script for downloading all Swarm CT 2-Hz files from Swarm FTP site
# As of 2021-02-24 neither wget nor ftplib work for downloading from Swarm's FTP site. Wonder if I'm banned ...
#
# Spencer Mark Hatch
# Birkeland Centre for Space Science
# 2021-01-18

import wget
from glob import glob
filelistdir = '/SPENCEdata/Research/SHEIC/filelists/'

swarmFTPAddr = "swarm-diss.eo.esa.int"
subfolder = 'New_baseline' 

DownloadDir = '/SPENCEdata/Research/database/Swarm/2Hz_TII_Cross-track/'
wantext = '.ZIP'
VERSION = '0302'

########################################
# Imports

from datetime import datetime 
import numpy as np
import pandas as pd
import os

########################################
# Define which satellites we'll look for during which years 

sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']
# y0,y1 = 2013,2021
# years = [str(val) for val in np.arange(y0,y1)]

########################################
# Download!
for sat in sats:

    print(sat)

    localdir = DownloadDir+'/'.join([sat.replace('Sat_','Swarm_'),''])
    if not os.path.exists(localdir):
        print(f"Making {localdir}")
        os.makedirs(localdir)

    ########################################
    ########################################
    # SINCE FTPLIB DOESN'T WORK, TRY THIS

    # Read in manually snatched list of files

    print("20210224 Even my new wget-based method doesn't work. The FTP site seems to hate me now, and I don't know why.")

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
    
