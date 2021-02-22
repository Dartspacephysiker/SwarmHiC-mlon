# try polarsub
import DAG.src.pysymmetry.visualization.polarsubplot
import pandas as pd
import numpy as np
import importlib
importlib.reload(DAG.src.pysymmetry.visualization.polarsubplot)
from DAG.src.pysymmetry.visualization.polarsubplot import Polarsubplot
plottrack_kws = dict(marker='.',SCALE=500,markercolor='blue',color='blue',markersize=10)
WEIMER_kws = dict(marker='.',SCALE=500,markercolor='red',color='red',markersize=10)

########################################
# MPL stuff
import matplotlib as mpl
import tkinter
mplBkgrnd = 'TkAgg'
mpl.use(mplBkgrnd)
mpl.rcParams.update({'text.color': 'k'})
mpl.rcParams.update({'axes.labelcolor': 'k'})
mpl.rcParams.update({'xtick.color': 'k'})
mpl.rcParams.update({'ytick.color': 'k'})
mpl.rcParams.update({'font.size': 15})
mpl.rcParams.update({'font.family': 'sans-serif'})
mpl.rcParams.update({'font.sans-serif': 'Arial'})
mpl.rcParams.update({'text.usetex': False})

mpl.rcParams.update({'figure.figsize': [10.0, 8.0]})
# mpl.rcParams.update({'savefig.directory': plotdir})

import matplotlib.pyplot as plt
plt.ion()

# fig,ax = plt.subplots(1,1); ax.hist2d(df[goodinds]['Viy_d1'],df[goodinds]['ViyWeimer_d1'],bins=[np.arange(-2000,2001,50),np.arange(-1500,1501,50)])

# fig,ax = plt.subplots(1,1); ax.hist2d(df[goodinds]['Viy_d1'],df[goodinds]['ViyWeimer_d1'],
#                                       bins=[np.arange(-2000,2001,50),
#                                             np.arange(-1500,1501,50)],
#                                       # cmax=5000,
#                                       norm=mpl.colors.LogNorm())
# ax.set_aspect('equal')

# fig,ax = plt.subplots(1,1); ax.scatter(df[goodinds]['Viy_d1'],df[goodinds]['ViyWeimer_d1'],marker='.',alpha=0.05)
# ax.set_xlim((-2500,2500))

def getinds(df,lilslice,only_calibrated=True):

    inds = pd.Series(np.zeros(df.shape[0],dtype=np.bool),index=df.index)
    inds.iloc[lilslice] = True

    if only_calibrated:
        goodinds = np.isfinite(df['Viy_d2']) & np.isfinite(df['ViyWeimer_d2']) & (df['Quality_flags'] == 4)
        inds = inds & goodinds

    return inds


# def showhim(showtime0,timedelta=pd.Timedelta('20 min')):

#     showtime1 = showtime0 + timedelta
#     showinds = (df.index >= showtime0) & (df.index <= showtime1)

#     fig = plt.figure(figsize=(15,9))
#     ax = fig.add_subplot(1,1,1)
#     # _ = ax.set_title(plottitle)

#     # _ = fig.suptitle(plottitle)

#     pax = Polarsubplot(ax,
#                        minlat = 45,
#                        linestyle="-",
#                        linewidth = 1,
#                        color = "lightgrey")
#     _ = pax.plot(df[showinds]['mlat'].values,
#                  df[showinds]['mlt'].values,
#                  color='black',
#                  alpha=1.0,
#                  linestyle='--')
#     # if doMagComponents:
#     pax.plottrack(df[showinds]['mlat'].values,
#                   df[showinds]['mlt'].values,
#                   df[showinds]['Viy_d2'].values*(-1),
#                   df[showinds]['Viy_d1'].values,**plottrack_kws)

#     _ = pax.plottrack(df[showinds]['mlat'].values,
#                       df[showinds]['mlt'].values,
#                       df[showinds]['ViyWeimer_d2'].values*(-1),
#                       df[showinds]['ViyWeimer_d1'].values,
#                       **WEIMER_kws)


def showhim(df,showinds,sat='Sat_A',
            northcol='Viy_d2',
            eastcol='Viy_d1'):#,
            # showtime0,timedelta=pd.Timedelta('20 min'),
            # showinds=None):

    # if showinds is None:

    #     showtime1 = showtime0 + timedelta
    #     showinds = (df.index >= showtime0) & (df.index <= showtime1)

    if np.sum(showinds) == 0:
        print("No indices! Returning ...")
        return

    fig = plt.figure(figsize=(10,9))
    ax = fig.add_subplot(1,1,1)
    # _ = ax.set_title(plottitle)

    # _ = fig.suptitle(plottitle)

    t0 = df[showinds].index[0]
    t1 = df[showinds].index[-1]
    hemi = 'NH' if ((df[showinds]['mlat']/np.abs(df[showinds]['mlat'])).median() > 0) else 'SH'
    titlestr = f"{sat.replace('Sat_','Swarm ')}, {hemi}\n[{t0.strftime('%Y-%m-%d %H:%M:%S')}, {t1.strftime('%Y-%m-%d %H:%M:%S')}]"
    print(titlestr)
    _ = ax.set_title(titlestr)

    multfac_d2 = -1 if (hemi == 'NH') else 1
    multfac_d2 = -1

    pax = Polarsubplot(ax,
                       minlat = 45,
                       linestyle="-",
                       linewidth = 1,
                       color = "lightgrey")
    _ = pax.plot(df[showinds]['mlat'].values,
                 df[showinds]['mlt'].values,
                 color='black',
                 alpha=1.0,
                 linestyle='--')
    # if doMagComponents:
    pax.plottrack(df[showinds]['mlat'].values,
                  df[showinds]['mlt'].values,
                  df[showinds][northcol].values*(multfac_d2),
                  df[showinds][eastcol].values,**plottrack_kws)

    _ = pax.plottrack(df[showinds]['mlat'].values,
                      df[showinds]['mlt'].values,
                      df[showinds]['ViyWeimer_d2'].values*(multfac_d2),
                      df[showinds]['ViyWeimer_d1'].values,
                      **WEIMER_kws)


def showhimscatter(df,showinds,
                   #showtime0,timedelta=pd.Timedelta('20 min'),
                   norm=None):

    goodinds = np.isfinite(df['Viy_d1']) & np.isfinite(df['ViyWeimer_d1']) \
        & np.isfinite(df['Viy_d2']) & np.isfinite(df['ViyWeimer_d2']) \
        & (df['Quality_flags'] == 4)
    # if showinds is None:

    #     showtime1 = showtime0 + timedelta
    #     showinds = (df.index >= showtime0) & (df.index <= showtime1) & goodinds

    showinds = showinds & goodinds

    if np.sum(showinds) == 0:
        print("No indices! Returning ...")
        return

    fig,axes = plt.subplots(1,2,figsize=(16,9),sharex=True,sharey=True);
    _ = axes[0].hist2d(df[showinds]['Viy_d1'],
                       df[showinds]['ViyWeimer_d1'],
                       bins=[np.arange(-600,601,20),
                             np.arange(-600,601,20)],
                       norm=norm)
    _ = axes[1].hist2d(df[showinds]['Viy_d2'],
                       df[showinds]['ViyWeimer_d2'],
                       bins=[np.arange(-600,601,20),
                             np.arange(-600,601,20)],
                       norm=norm)

    _ = axes[0].set_xlabel("Swarm A cross-track convection [m/s]")
    _ = axes[1].set_xlabel("Swarm A cross-track convection [m/s]")
    _ = axes[0].set_ylabel("Weimer cross-track convection [m/s]")

    axes[0].set_title("Apex $d_1$ (approx. E-W)")
    axes[1].set_title("Apex $d_2$ (approx. N-S)")

    return fig,axes

def get_passes(df):
    assert df.index.is_monotonic,"df index not monotonic!"

    deltats = np.diff(df.index)/np.timedelta64(1)/1e9
    shpasses = np.ma.clump_unmasked(np.ma.masked_greater(df['mlat'].values,-45))
    nhpasses = np.ma.clump_unmasked(np.ma.masked_less(df['mlat'].values,45))

    return deltats,nhpasses,shpasses


def smooth_df_Viy(df,window='10 s',center=False):
    # if inplace:

    if center:
        viyd1roll = df['Viy_d1'].rolling(window).mean()
        viyd2roll = df['Viy_d2'].rolling(window).mean()

        viyd1roll.index = viyd1roll.index.shift(-1,freq=pd.Timedelta(window)/2)
        viyd2roll.index = viyd2roll.index.shift(-1,freq=pd.Timedelta(window)/2)


        viyd1roll = viyd1roll.reindex(viyd1roll.index.union(df['Viy_d1'].index)).interpolate(method = 'linear', limit = 30)
        viyd2roll = viyd2roll.reindex(viyd2roll.index.union(df['Viy_d2'].index)).interpolate(method = 'linear', limit = 30)

        df['Viy_d1roll'] = viyd1roll
        df['Viy_d2roll'] = viyd2roll

    else:
        df['Viy_d1roll'] = df['Viy_d1'].rolling(window).mean()
        df['Viy_d2roll'] = df['Viy_d2'].rolling(window).mean()

    # else:
    #     dfcopy = df.copy()
    #     dfcopy['Viy_d1roll'] = dfcopy['Viy_d1'].rolling(window).mean()
    #     dfcopy['Viy_d2roll'] = dfcopy['Viy_d2'].rolling(window).mean()
        
