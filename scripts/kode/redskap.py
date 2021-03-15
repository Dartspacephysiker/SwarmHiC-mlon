#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:24:49 2021

@author: qew008
"""

import pandas as pd
import numpy as np

def konveksjonsdatabase(sat,bare_substorm=False):
    #%%
    #Laste inn pakker, velge satellitt
    
    datadir = '/Data/ift/ift_romfys1/Q1/folk/anna_kvamsdal/data/'
    hdffile = f'Sat_{sat}_ct2hz_v0302_5sres.h5'
    
    #%%
    #Se HDFs innehold
    
    # print(f"Opening {hdffile}")
    # with pd.HDFStore(datadir+hdffile,'r') as store:
    #     print(store.keys())
    #     print("'/external' member contains:")
    #     print(store['/external'].columns)
        
    #%%
    #Velge ut noen kolonner og lage en DataFrame
    getcols = ['/mlt','/mlat','/alt','/yhat_d1','/yhat_d2','/Viy_d1','Viy_d2','/Quality_flags']
    getallextcols = True
    
    df = pd.DataFrame()
    with pd.HDFStore(datadir+hdffile,'r') as store:
        print("Getting these columns: "+", ".join(getcols)+' ...',end='')
        for col in getcols:
            df[col.replace('/','')] = store[col]
        print("Done!")
    
        if getallextcols:
            print("Getting solar wind, IMF, dipoletilt, F10.7, and substorm data ...",end='')
    
            df = df.join(store['/external'])
            dfcols = list(df.columns)
            renamecols = dict(Bz='IMFBz',By='IMFBy')
            for key,rcol in renamecols.items():
                dfcols[np.where([key == col for col in dfcols])[0][0]] = rcol
            df.columns = dfcols
            print("Done!")
    #%%
    #Fjerne alle rader som ikke har Quality_flags >= 4 (disse er dårlig kalibrert)
    print("Junking data with Quality_flags < 4 ... ",end='')
    N = df.shape[0]
    good = df['Quality_flags'] >= 4
    df = df[good]
    print(f"Junked {N - df.shape[0]} rows and kept {df.shape[0]}")
    
    if bare_substorm:
        print("Only keeping rows associated with finite substorm onset parameters ...",end='')
        N = df.shape[0]
        good = np.isfinite(df['onset_mlt'])
        df = df[good]
        print(f"Junked {N - df.shape[0]} rows and kept {df.shape[0]}") 

        
    return df
