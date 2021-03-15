#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:57:21 2021

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

#%% Lage grid
griddmlat = 2
griddmlon = 0.75
grid, forstermlts, forstermltres = sdarngrid(latmin=50,dlat=griddmlat,dlon=griddmlon,return_mltres=True)

gridmlats = grid[0,:]+griddmlat/2.
gridmlts = forstermlts+forstermltres/2.
nGridPts = len(gridmlats)

#%% Lage plott som viser litt kunstig data

# Lage kunstig data
fakedataN = np.cos(np.deg2rad((forstermlts+forstermltres/2)*15))
fakedataS = np.sin(np.deg2rad((forstermlts+forstermltres/2)*15))

fakedata = [fakedataN,fakedataS]
axnames = ['North','South']
fakedatanames = ['Cos(MLT*15)',"Sin(MLT*15)"]

# min/maks kolorbarverdier
vmin,vmax = -1,1
normer=mpl.colors.Normalize(vmin=vmin,vmax=vmax)

ncols=2
colwidth = 10
rowwidth=20

cmap = 'summer'
cmap = 'spring'

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
        fakedata[iax],
        norm=normer,
        cmap=cmap)# ,
    
    paxes.append(pax)

    cbs.append(plt.colorbar(mpl.cm.ScalarMappable(norm=normer, cmap=cmap),cax=caxes[iax],
                 orientation='horizontal',
                 pad=0.2,
                 shrink=0.5))
    cbs[iax].set_label(fakedatanames[iax])
    
#%% Gridded data
    
# tmpbins = bin_number(tmpdf[latlabel].values,
#                                         tmpdf[LTlabel].values,
#                                         forstermlat,forsterdmlat,
#                                         forstermlt,forsterdmlt)
