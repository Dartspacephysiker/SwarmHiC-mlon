#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:35:50 2021

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
plotdir = '/Data/ift/ift_romfys1/Q1/folk/anna_kvamsdal/plott/substorm_info/'
#%% Først må du kjøre åpne_databasen.py.

#%% Finne ut av hvor mange målinger med hver satellitt

#df[df['sat'] == 'A'].shape[0]
#[776343 rows x 19 columns]

#df[df['sat'] == 'B'].shape[0]
# [651997 rows x 19 columns]

#Altså 1,4 M til sammen

#%%% viktige parametere
indsN = df['mlat'] > 0
indsS = df['mlat'] < 0

dfhemi = df[indsN]

Byposinds = dfhemi['IMFBy'] > 0
Byneginds = dfhemi['IMFBy'] < 0

#20-22h, early
earlyinds = (dfhemi["onset_mlt"] > 20) & (dfhemi["onset_mlt"] < 22)

Byposearlyinds = Byposinds & earlyinds
Bynegearlyinds = Byneginds & earlyinds

#1-3h, late
lateinds = (dfhemi['onset_mlt'] > 1) & (dfhemi['onset_mlt'] < 3)

Byposlateinds = Byposinds & lateinds
Byneglateinds = Byneginds & lateinds

#%%Få tilbake "ekte" substorm-databasen når den kommer

ssuniqinds = ~dfhemi[['onset_mlat', 'onset_mlt']].duplicated()
ss = dfhemi[ssuniqinds][['tilt','onset_dt_minutes','onset_mlat', 'onset_mlt', "IMFBy", "IMFBz"]]
onsetdt = pd.TimedeltaIndex(-ss['onset_dt_minutes']*60,unit='s')
ss = ss.set_index(ss.index+onsetdt)

#%% Antall målinger innen ulike bin.

#Velger halvkule i seksjonen: viktiger parametere.
dfhemi['bin_number'] = np.int64(99999) #skaper bin kollonen

dfhemi['bin_number'] = bin_number(grid,dfhemi['mlat'].values,dfhemi['mlt'].values)

# dfN['bin_number'] = bin_number_spence(dfN['mlat'].values,dfN['mlt'].values,forster)
# dfS['bin_number'] = bin_number(grid,np.abs(dfS['mlat'].values),dfS['mlt'].values)

# bin_number_spence(mlat,mlt,
#                       forstermlat,forsterdmlat,
#                       forstermlt,forsterdmlt,
#                       verbose=False)

#Hvor mange målinger i hvert bin? Må få skrivd opp de fire mulighetene!??
count_Byposearly = dfhemi[Byposearlyinds].groupby('bin_number')["sat"].count()
count_Bynegearly = dfhemi[Bynegearlyinds].groupby('bin_number')["sat"].count()
count_Byposlate = dfhemi[Byposlateinds].groupby('bin_number')["sat"].count()
count_Byneglate = dfhemi[Byneglateinds].groupby('bin_number')["sat"].count()



# Lage kunstig data

countdata = [countN, countS]
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

gs = fig.add_gridspec(rowwidth, ncols*colwidth)
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
#%% Binne data

#%% UT-histogram
#Få klokkeslett til hver substorm
UTdfhemi = dfhemi.index.hour.values+dfhemi.index.minute.values/60+dfhemi.index.second.values/3600

fig,axes = plt.subplots(2,2)
fig.suptitle("UT", fontsize=16)
axes = axes.flatten()


axes[0].hist(UTdfhemi[Byposearlyinds])
axes[0].set_title("By > 0")
axes[0].set_ylabel("Early")
axes[1].hist(UTdfhemi[Bynegearlyinds])
axes[1].set_title("By < 0")
axes[2].hist(UTdfhemi[Byposlateinds])
axes[2].set_ylabel("Late")
axes[3].hist(UTdfhemi[Byneglateinds])


#%% tilt-histogram
#Ca samme som UT-histogram

fig,ax = plt.subplots(1,1)
ax.hist(dfhemi.tilt)
ax.set_xlabel("tilt")

#%% N onset_dt_minutes:
fig,axes = plt.subplots(2,2)
fig.suptitle("t_onset", fontsize=16)
axes = axes.flatten()

axes[0].hist(dfhemi[Byposearlyinds].onset_dt_minutes)
axes[0].set_title("By > 0")
axes[0].set_ylabel("Early")
axes[1].hist(dfhemi[Bynegearlyinds].onset_dt_minutes)
axes[1].set_title("By < 0")
axes[2].hist(dfhemi[Byposlateinds].onset_dt_minutes)
axes[2].set_ylabel("Late")
axes[3].hist(dfhemi[Byneglateinds].onset_dt_minutes)

#%% Substorm mlat/MLT scatterplot

fig,ax = plt.subplots(1,1)
#ax.scatter(ss.onset_mlat, ss.onset_mlt)
pax = Polarsubplot(ax)
pax.scatter(dfhemi.onset_mlat, dfhemi.onset_mlt, marker=".", alpha=0.05)
ax.set_title('Substorm onset')
#%% Deler opp målingene etter By


fig,axes = plt.subplots(2,2)
fig.suptitle("Antall målinger for ulik By", fontsize=16)
axes = axes.flatten()
paxes = [Polarsubplot(ax) for ax in axes]

paxkw = dict(marker='x',alpha=0.005)

paxes[0].scatter(dfhemi[Byposearlyinds].onset_mlat, dfhemi[Byposearlyinds].onset_mlt,**paxkw)
axes[0].set_title("By > 0")
#axes[0].set_ylabel("Early")
paxes[1].scatter(dfhemi[Bynegearlyinds].onset_mlat, dfhemi[Bynegearlyinds].onset_mlt,**paxkw)
axes[1].set_title("By < 0")
paxes[2].scatter(dfhemi[Byposlateinds].onset_mlat, dfhemi[Byposlateinds].onset_mlt,**paxkw)
#axes[2].set_ylabel("Late")
paxes[3].scatter(dfhemi[Byneglateinds].onset_mlat, dfhemi[Byneglateinds].onset_mlt,**paxkw)