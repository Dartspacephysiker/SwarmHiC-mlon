#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:35:50 2021

@author: qew008
"""
from polarsubplot import Polarsubplot
from grids import sdarngrid,bin_number

import matplotlib as mpl
mpl.rcParams.update({'text.usetex': False})

import matplotlib.pyplot as plt


plotdir = '/Data/ift/ift_romfys1/Q1/folk/anna_kvamsdal/plott/substorm_info/'
#%% Først må du kjøre åpne_databasen.py.

#%% Finne ut av hvor mange målinger med hver satellitt

# df[df['sat'] == 'A'].shape[0]
#[776343 rows x 19 columns]

# df[df['sat'] == 'B'].shape[0]
# [651997 rows x 19 columns]

#Altså 1,4 M til sammen


#%%Få tilbake "ekte" substorm-databasen når den kommer

ssuniqinds = ~df[['onset_mlat', 'onset_mlt']].duplicated()
ss = df[ssuniqinds][['tilt','onset_dt_minutes','onset_mlat', 'onset_mlt','IMFBy','IMFBz']]
onsetdt = pd.TimedeltaIndex(-ss['onset_dt_minutes']*60,unit='s')
ss = ss.set_index(ss.index+onsetdt)

#%% UT-histogram
#Få klokkeslett til hver substorm
UTss = ss.index.hour.values+ss.index.minute.values/60+ss.index.second.values/3600


fig,ax = plt.subplots(1,1)
ax.hist(UTss)
ax.set_xlabel("Substorm UT")
plt.savefig(plotdir+"UThist.png")

#%% tilt-histogram
#Ca samme som UT-histogram


plt.savefig(plotdir+"tilthist.png")
#%% Substorm mlat/MLT scatterplot
fig,ax = plt.subplots

#pax = Polarsubplot(ax)
#pax.scatter


#%% Dele opp målinger etter By

Byposinds = ss['IMFBy'] > 0
Byneginds = ss['IMFBy'] < 0

lateinds = (ss['onset_mlt'] > 23) | (ss['onset_mlt'] < 6)

Byposlateinds = Byposinds & lateinds
Byneglateinds = Byneginds & lateinds

ss[Byposlateinds].onset_mlt

fig,axes = plt.subplots(2,2)
axes = axes.flatten()
paxes = [Polarsubplot(ax) for ax in axes]

paxkw = dict(marker='x',alpha=0.1)

paxes[0].scatter(,**paxkw)
paxes[1].scatter(,**paxkw)
paxes[2].scatter(,**paxkw)