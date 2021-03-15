#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 11:12:09 2021

@author: qew008
"""

import pandas as pd
import numpy as np
from redskap import konveksjonsdatabase
from polarsubplot import Polarsubplot
from grids import sdarngrid,bin_number

import matplotlib as mpl
mpl.rcParams.update({'text.usetex': False})

import matplotlib.pyplot as plt

#%% Laste inn database

# dbopts = dict(bare_substorm=True)

# sats = ['A','B']
# df = []
# print("Getting all convection measurements associated with a substorm ...")
# for sat in sats:
#     df.append(konveksjonsdatabase(sat,**dbopts))
#     df[-1]['sat'] = sat
# df = pd.concat(df)


#%% Lage grid
griddmlat = 2
griddmlon = 3
grid, forstermlts, forstermltres = sdarngrid(latmin=45,dlat=griddmlat,dlon=griddmlon,return_mltres=True)

gridmlats = grid[0,:]+griddmlat/2.
gridmlts = forstermlts+forstermltres/2.
nGridPts = len(gridmlats)


#%% Binne data

#Dele inn in NH- og SH-målinger

indsN = df['mlat'] > 0
indsS = df['mlat'] < 0

df['bin_number'] = np.int64(99999)
dfN = df[indsN]
dfS = df[indsS]

dfN['bin_number'] = bin_number(grid,dfN['mlat'].values,dfN['mlt'].values)
dfS['bin_number'] = bin_number(grid,np.abs(dfS['mlat'].values),dfS['mlt'].values)

# dfN['bin_number'] = bin_number_spence(dfN['mlat'].values,dfN['mlt'].values,forster)
# dfS['bin_number'] = bin_number(grid,np.abs(dfS['mlat'].values),dfS['mlt'].values)

# bin_number_spence(mlat,mlt,
#                       forstermlat,forsterdmlat,
#                       forstermlt,forsterdmlt,
#                       verbose=False)

#Hvor mange målinger i hvert bin?
countN = dfN.groupby('bin_number')['sat'].count()
countS = dfS.groupby('bin_number')['sat'].count()

#%% Lage plott som viser litt kunstig data

# Lage kunstig data

countdata = [countN,countS]
axnames = ['North','South']
fakedatanames = ['Count','Count']

# min/maks kolorbarverdier
vmin,vmax = -1,1
normN =mpl.colors.Normalize(vmin=countN.min(),vmax=countN.max())
normS =mpl.colors.Normalize(vmin=countS.min(),vmax=countS.max())

norms = [normN,normS]

ncols=2
colwidth = 10
rowwidth=20

cmap = 'summer'
cmap = 'plasma'

fig = plt.figure(figsize=(10,5))
#axN = fig.add_subplot(1,2,1)
#axS = fig.add_subplot(1,2,2)
#axes = [axN,axS]

gs = fig.add_gridspec(rowwidth+1, ncols*colwidth)
ax00 = fig.add_subplot(gs[:rowwidth, :colwidth])
ax01 = fig.add_subplot(gs[:rowwidth, colwidth:colwidth*2])
axes = [ax00,ax01]
cax00 = fig.add_subplot(gs[rowwidth:,:colwidth])
cax01 = fig.add_subplot(gs[rowwidth:,colwidth:colwidth*2])
axes = [ax00,ax01]
caxes = [cax00,cax01]


paxes = []
cbs = []
for iax,ax in enumerate(axes):
    
    ax.set_title(axnames[iax])
    pax = Polarsubplot(ax,linestyle='--',color='gray',minlat=45)
    

    bro = pax.filled_cells(
        grid[0,:],forstermlts,griddmlat,forstermltres,
        countdata[iax],
        norm=norms[iax],
        cmap=cmap)# ,
    
    paxes.append(pax)

    cbs.append(plt.colorbar(mpl.cm.ScalarMappable(norm=norms[iax], cmap=cmap),
                            cax=caxes[iax],
                 orientation='horizontal',
                 pad=0.2,
                 shrink=0.5))
    cbs[iax].set_label(fakedatanames[iax])
    
#%% Gridded data
    
# tmpbins = bin_number(tmpdf[latlabel].values,
#                                         tmpdf[LTlabel].values,
#                                         forstermlat,forsterdmlat,
#                                         forstermlt,forsterdmlt)
