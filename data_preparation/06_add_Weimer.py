# import importlib
# import hatch_python_utils.satellites.Swarm_TII
# importlib.reload(hatch_python_utils.satellites.Swarm_TII)
from pysymmetry import geodesy
import pandas as pd
import numpy as np
from hatch_python_utils.math.vectors import dotprod

from hatch_python_utils.time_tools import doy_single    

# import DAG.src.pysymmetry.models.weimer.convection
# importlib.reload(DAG.src.pysymmetry.models.weimer.convection)
from DAG.src.pysymmetry.models.weimer.convection import Convection

# Make vectorizable version of weimer05 
weimer = np.vectorize(lambda iyear,iday,uthr,mlat,mlt,by,bz,swvel,swden: wei05sc.weimer05(iyear,iday,uthr,mlat,mlt,by,bz,swvel,swden))

from hatch_python_utils.satellites.Swarm_TII import calculate_satellite_frame_vectors_in_NEC_coordinates,calculate_crosstrack_flow_in_NEC_coordinates

def get_Weimer_convec_in_crosstrack_direction_in_apex_coords(df,vNWeimer,vEWeimer):
    """
    Name says it all!
    """
    # Satellite frame unit vectors in NEC coords
    xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=['VsatN','VsatE','VsatC'])

    # Satellite-frame y unit vector in Apex coords
    yhat_d1 = df['d10']*yhat[1,:] + df['d11']*yhat[0,:] + df['d12']*yhat[2,:]
    yhat_d2 = df['d20']*yhat[1,:] + df['d21']*yhat[0,:] + df['d22']*yhat[2,:]
    yhat_apex = np.array([yhat_d2*(-1),yhat_d1,yhat_d1*0])

    # Now dot Weimer vector (which is in AACGM/Apex coords) with yhat_apex vector to get magnitude of Weimer flow in cross-track direction
    vWeimer_yhat = dotprod(yhat_apex.T,np.array([vNWeimer,vEWeimer,vEWeimer*0]).T)

    return vWeimer_yhat


datapath = '/SPENCEdata/Research/database/SHEIC/'
# storefn = '/SPENCEdata/Research/database/SHEIC/data_v1_update.h5'
# groups = ['SwarmA', 'SwarmB', 'SwarmC']

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A']
VERSION = '0302'
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
hdfsuff = '_NOWDAT'
hdfsuff = '_Anna'
hdfsuff = '_Anna2'

OMNIPERIOD = '20Min'

getcols = ['Bx','By','Bz',
           'Quality_flags',
           'Latitude','Longitude','Radius',
           'Viy',
           'mlat','mlon','mlt',
           'VsatN','VsatE','VsatC',
           'd10','d11','d12',
           'd20','d21','d22']

##############################
# Make df with getcols
df = pd.DataFrame()

for sat in sats:
    print(sat)

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(masterhdf)

    with pd.HDFStore(masterhdfdir+masterhdf, 'r') as store:
        times = store.select_column('/mlt', 'index').values # the time index

        for wantcol in getcols:
            df[wantcol] = store['/'+wantcol]

y1 = str(df.index.min().year)
y2 = str(df.index.max().year)

##############################
# Load OMNI
omnicols = dict(bz = 'BZ_GSM',
                by = 'BY_GSM',
                vx = 'Vx',
                nsw='proton_density')
with pd.HDFStore(datapath + 'omni_1min.h5', 'r') as omni:
    # ORIG
    # Bz = omni['/omni'][omnicols['bz']][y1:y2].rolling(OMNIPERIOD).mean()
    # By = omni['/omni'][omnicols['by']][y1:y2].rolling(OMNIPERIOD).mean()
    # vx = omni['/omni'][omnicols['vx']][y1:y2].rolling(OMNIPERIOD).mean()

    # NY
    external = omni['/omni']
    external = external[~external.index.duplicated(keep='first')]
    Bz = external[omnicols['bz']][y1:y2].rolling(OMNIPERIOD).mean()
    By = external[omnicols['by']][y1:y2].rolling(OMNIPERIOD).mean()
    vx = external[omnicols['vx']][y1:y2].rolling(OMNIPERIOD).mean()
    nsw = external[omnicols['nsw']][y1:y2].rolling(OMNIPERIOD).mean()

external = pd.DataFrame([Bz, By, vx, nsw]).T
external.columns = ['Bz', 'By', 'vx', 'nsw']
external = external.dropna()


##############################
# Get convection in geodetic and Apex coordinates
# Bonus: get altitude and geodetic latitude
Viy_NEC = calculate_crosstrack_flow_in_NEC_coordinates(df)

df['Viy_N'] = Viy_NEC[0,:]
df['Viy_E'] = Viy_NEC[1,:]
df['Viy_C'] = Viy_NEC[2,:]

# Convert 'Viy_N' (and 'Viy_C') from geocentric to geodetic coordinates
gdlat, alt, Viy_NGeod, Viy_DGeod = geodesy.geoc2geod(90.-df["Latitude"].values,
                                                     df["Radius"].values/1000.,
                                                     -df['Viy_N'],df['Viy_C'])
                                     # -df.B_N, -df.B_C)

# Now Apex coords
# REMEMBER (from Richmond, 1995):
# • d1 base vector points "more or less in the magnetic eastward direction"
# • d2 base vector points "generally downward and/or equatorward (i.e., southward in NH [and in SH?])" 
df['Viy_d1'] = df['d10']*df['Viy_E'] + df['d11']*Viy_NGeod + df['d12']*Viy_DGeod
df['Viy_d2'] = df['d20']*df['Viy_E'] + df['d21']*Viy_NGeod + df['d22']*Viy_DGeod

df['gdlat'] = gdlat
df['alt'] = alt


# Also get satellite yhat vector in Apex coordinates
xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=['VsatN','VsatE','VsatC'])
yhat_d1 = df['d10']*yhat[1,:] + df['d11']*yhat[0,:] + df['d12']*yhat[2,:]
yhat_d2 = df['d20']*yhat[1,:] + df['d21']*yhat[0,:] + df['d22']*yhat[2,:]

##############################
# Get Weimer values

df['Bmag'] = np.sqrt((df[['Bx','By','Bz']]**2).sum(axis=1))


# Loop over stuff with resolution 30 s
dt = '180 s'
maxtdiff_between_wanttime_and_omni = pd.Timedelta('2 min')
td = pd.Timedelta(dt)
tmin = df.index.min()
tmax = df.index.max()
times = pd.date_range(tmin,tmax,freq=dt) 

df['ViyWeimer_d1'] = np.nan
df['ViyWeimer_d2'] = np.nan

for it,t in enumerate(times):
    print(f"{it:4d} {t.strftime('%Y-%m-%d %H:%M:%S')}",end=' ')
    hereinds = (df.index >= t) & (df.index < (t + td))
    nhere = hereinds.sum()
    if nhere == 0:
        print("Skip!")
        continue
    print(nhere)


    # Get Weimer params for this moment

    nearestind = np.abs(external.index-t).argsort()[0]
    if np.abs(external.iloc[nearestind].name-t) > maxtdiff_between_wanttime_and_omni:
        continue

    iyear = t.year
    iday = np.int64(np.floor(doy_single(t)))
    uthr = t.hour + (t.minute/60. + t.second/3600.)/24.

    Bz, By, vsw, nsw = external.iloc[nearestind].values
    vsw = np.abs(vsw)           # Weimer model wants it to be abs
    
    print(f"{iyear} {iday} {uthr:6.2f} {By:6.2f} {Bz:6.2f} {vsw:8.2f} {nsw:7.2f}")

    # vNWeimer, vEWeimer, vBWeimer = weimer_convection(iyear,iday,uthr,by,bz,vsw,nsw)
    vNWeimer, vEWeimer, vBWeimer = Convection(df[hereinds]['mlat'].values,df[hereinds]['mlt'].values,
                                              df[hereinds]['Bmag'].values,
                                              iyear,iday,uthr,By,Bz,vsw,nsw)
    
    vWeimer_yhat = get_Weimer_convec_in_crosstrack_direction_in_apex_coords(df[hereinds],vNWeimer,vEWeimer)

    df.loc[hereinds,'ViyWeimer_d1'] = vWeimer_yhat*yhat_d1[hereinds]
    df.loc[hereinds,'ViyWeimer_d2'] = vWeimer_yhat*yhat_d2[hereinds]
    
with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
    for column in ['ViyWeimer_d1','ViyWeimer_d2']:
        store.append(column, df[column], format='t')

# goodinds = np.isfinite(df['Viy_d2']) & np.isfinite(df['ViyWeimer_d2'])

# goodinds = np.isfinite(df['Viy_d2']) & np.isfinite(df['ViyWeimer_d2']) & (df['Quality_flags'] == 4)


assert df.index.is_monotonic

deltats = np.diff(df.index)/np.timedelta64(1)/1e9
shpasses = np.ma.clump_unmasked(np.ma.masked_greater(df['mlat'].values,-45))
nhpasses = np.ma.clump_unmasked(np.ma.masked_less(df['mlat'].values,45))

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

# try polarsub
import DAG.src.pysymmetry.visualization.polarsubplot
import importlib
importlib.reload(DAG.src.pysymmetry.visualization.polarsubplot)
from DAG.src.pysymmetry.visualization.polarsubplot import Polarsubplot
plottrack_kws = dict(marker='.',SCALE=500,markercolor='blue',color='blue',markersize=10)
WEIMER_kws = dict(marker='.',SCALE=500,markercolor='red',color='red',markersize=10)

def showhim(showtime0,timedelta=pd.Timedelta('20 min')):

    showtime1 = showtime0 + timedelta
    showinds = (df.index >= showtime0) & (df.index <= showtime1)

    fig = plt.figure(figsize=(15,9))
    ax = fig.add_subplot(1,1,1)
    # _ = ax.set_title(plottitle)
    
    # _ = fig.suptitle(plottitle)
    
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
                  df[showinds]['Viy_d2'].values*(-1),
                  df[showinds]['Viy_d1'].values,**plottrack_kws)
    
    _ = pax.plottrack(df[showinds]['mlat'].values,
                      df[showinds]['mlt'].values,
                      df[showinds]['ViyWeimer_d2'].values*(-1),
                      df[showinds]['ViyWeimer_d1'].values,
                      **WEIMER_kws)

showtime0 = pd.Timestamp('2016-01-28 10:50:00'); showhim(showtime0,timedelta=pd.Timedelta('30 min'))
