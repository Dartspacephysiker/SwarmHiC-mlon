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

binnr = 100
print(f"bin {binnr} mlat/mlt: {gridmlats[binnr]:.2f}/{gridmlts[binnr]:.2f}")

binindsN = dfN["bin_number"] == binnr
print(dfN[binindsN][["mlat", "mlt", "yhat_d1", "yhat_d2"]].describe())

fig, ax = plt.subplots(1,1)
d1, d2 = dfN[binindsN][["yhat_d1","yhat_d2"]].values.T
ax.scatter(d1,d2)
# dfN['bin_number'] = bin_number_spence(dfN['mlat'].values,dfN['mlt'].values,forster)
# dfS['bin_number'] = bin_number(grid,np.abs(dfS['mlat'].values),dfS['mlt'].values)

# bin_number_spence(mlat,mlt,
#                       forstermlat,forsterdmlat,
#                       forstermlt,forsterdmlt,
#                       verbose=False)

#Hvor mange målinger i hvert bin?
#countN = dfN.groupby('bin_number')['sat'].count()
#countS = dfS.groupby('bin_number')['sat'].count()


#%% Lage plott som viser litt kunstig data

yhatd1meanN = dfN.groupby("bin_number")["yhat_d1"].mean()
yhatd2meanN = dfN.groupby("bin_number")["yhat_d2"].mean()
yhatbiasN = np.sqrt(yhatd1meanN**2 + yhatd2meanN**2)

yhatd1meanS = dfS.groupby("bin_number")["yhat_d1"].mean()
yhatd2meanS = dfS.groupby("bin_number")["yhat_d2"].mean()
yhatbiasS = np.sqrt(yhatd1meanS**2 + yhatd2meanS**2)

# Lage kunstig data

countdata = [yhatbiasN, yhatbiasS]
axnames = ['North','South']
fakedatanames = ['Bias','Bias']

# min/maks kolorbarverdier
vmin,vmax = -1,1
normN =mpl.colors.Normalize(0,1)
normS =mpl.colors.Normalize(0,1)

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

#%% Bias-vinkel


dfN["yhd1abs"] = dfN['yhat_d1'].abs()
dfN["yhd2abs"] = dfN['yhat_d2'].abs()

dfS["yhd1abs"] = dfS['yhat_d1'].abs()
dfS["yhd2abs"] = dfS['yhat_d2'].abs()

yhatd1absmeanN = dfN.groupby("bin_number")["yhd1abs"].mean()
yhatd2absmeanN = dfN.groupby("bin_number")["yhd2abs"].mean()
yhatd1absmeanS = dfS.groupby("bin_number")["yhd1abs"].mean()
yhatd2absmeanS = dfS.groupby("bin_number")["yhd2abs"].mean()


# yhatd1absmeanN = np.abs(dfN.groupby("bin_number")["yhat_d1"]).mean()
# yhatd2absmeanN = np.abs(dfN.groupby("bin_number")["yhat_d2"]).mean()

# yhatd1absmeanS = np.abs(dfS.groupby("bin_number")["yhat_d1"]).mean()
# yhatd2absmeanS = np.abs(dfS.groupby("bin_number")["yhat_d2"]).mean()

yhatbiasvN = np.rad2deg(np.arctan2(yhatd2absmeanN,yhatd1absmeanN))
yhatbiasvS = np.rad2deg(np.arctan2(yhatd2absmeanS,yhatd1absmeanS))
#yhatbiasS = np.sqrt(yhatd1meanS**2 + yhatd2meanS**2)

# Lage kunstig data

countdata = [yhatbiasvN, yhatbiasvS]
axnames = ['North','South']
fakedatanames = ['Bias','Bias']

# min/maks kolorbarverdier
vmin,vmax = 0,90
normN =mpl.colors.Normalize(0,90)
normS =mpl.colors.Normalize(0,90)

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