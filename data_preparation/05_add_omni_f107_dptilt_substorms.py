import pandas as pd
import numpy as np
from dipole import dipole_tilt
from datetime import datetime

datapath = '/SPENCEdata/Research/database/SHEIC/'

doAddSubstorms = True
# Downloaded from https://supermag.jhuapl.edu/substorms/
ssinfile = '/SPENCEdata/Research/database/SHEIC/substorms-ohtani-20131201_000000_to_20211202_000000.ascii'
ss_maxdt_tolerance = '30 min'

# sats = ['Sat_A','Sat_B','Sat_C']
# sats = ['Sat_B','Sat_C']
sats = ['Sat_A']

VERSION = '0302'
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
hdfsuff = '_5sres'
hdfsuff = '_1sres'
# hdfsuff = '_Anna'
# hdfsuff = '_Anna2'
# hdfsuff = '_2014'


if hdfsuff == '_5sres':
    y1 = '2013'
    y2 = '2021'
if hdfsuff == '_1sres':
    y1 = '2013'
    y2 = '2021'
elif hdfsuff == '_Anna':
    y1 = '2013'
    y2 = '2014'
elif hdfsuff == '_Anna2':
    y1 = '2016'
    y2 = '2016'
elif hdfsuff == '_2014':
    y1 = '2014'
    y2 = '2015'


PERIOD = '20Min'

print("The plan: Add OMNI, F10.7, " \
      + ("and dipole tilt" if not doAddSubstorms else "dipole tilt, and substorm onset params") \
      + " to HDF files")

##############################
# OMNI

print("Load OMNI first ... ",end='')
omnicols = dict(bz = 'BZ_GSM',
                by = 'BY_GSM',
                vx = 'Vx',
                nsw='proton_density')
with pd.HDFStore(datapath + 'omni_1min.h5', 'r') as omni:
    # ORIG
    # Bz = omni['/omni'][omnicols['bz']][y1:y2].rolling(PERIOD).mean()
    # By = omni['/omni'][omnicols['by']][y1:y2].rolling(PERIOD).mean()
    # vx = omni['/omni'][omnicols['vx']][y1:y2].rolling(PERIOD).mean()

    # NY
    external = omni['/omni']
    external = external[~external.index.duplicated(keep='first')]
    Bz = external[omnicols['bz']][y1:y2].rolling(PERIOD).mean()
    By = external[omnicols['by']][y1:y2].rolling(PERIOD).mean()
    vx = external[omnicols['vx']][y1:y2].rolling(PERIOD).mean()
    nsw = external[omnicols['nsw']][y1:y2].rolling(PERIOD).mean()

external = pd.DataFrame([Bz, By, vx, nsw]).T
external.columns = ['Bz', 'By', 'vx', 'nsw']
external = external.dropna()

##############################
# F10.7

print("og så  F10.7 ... ",end='')
f107cols = dict(f107obs='observed_flux (solar flux unit (SFU))',
                f107adj='adjusted_flux (solar flux unit (SFU))')

f107 = pd.read_csv(datapath + 'penticton_radio_flux.csv', sep = ',', parse_dates= True, index_col = 0)  
f107[f107 == 0] = np.nan
# Convert Julian to datetime 
time = np.array(f107.index)
epoch = pd.to_datetime(0, unit = 's').to_julian_date()
time = pd.to_datetime(time-epoch, unit = 'D')
#set datetime as index
f107 = f107.reset_index()
f107.set_index(time, inplace=True)
f107 = f107[~f107.index.duplicated(keep='first')]
f107 = f107.sort_index()
f107 = f107[f107.index >= pd.Timestamp(y1+'-01-01')]

# interpolate f107: 
print("(interpolating F10.7 to match OMNI ...) ",end='')
f107[f107cols['f107obs']][f107[f107cols['f107obs']] < 0] = np.nan
f107[f107cols['f107adj']][f107[f107cols['f107adj']] < 0] = np.nan
# there is a huge data gap last 8 months of 2018 - I just inteprolate over this
f107 = f107.reindex(f107.index.union(external.index)).interpolate(method = 'linear', limit = 24*60*8*31)
for key,val in f107cols.items():
    external[key] = f107[val]

##############################
# Dipole tilt

print("then dipole tilt ... ",end='')
external['tilt'] = np.nan
for year in np.unique(external.index.year):
    print('calculting tilt for %s' % year)
    external.loc[str(year), 'tilt'] = dipole_tilt(external[str(year)].index, year)

##############################
# Substorms

if doAddSubstorms:    
    interper = lambda y,m,d,H,M: datetime.strptime(y+m+d+H+M,"%Y%m%d%H%M")
    #names=['mlt','mlat','glon','glat']
    names=['yr','mo','day','h','m','mlt','mlat','glon','glat']
    # read in substorm list
    ss = pd.read_csv(ssinfile,
                     sep='\s+',
                     skiprows=38,
                     header=None,
                     infer_datetime_format=True,
                     parse_dates={'time': [0,1,2,3,4]},
                     date_parser=interper,names=names)
    ss = ss.set_index('time')
    
    assert ss.index.is_monotonic,"You might run into trouble"
    
##############################
# Add data to master hdfs

for sat in sats:
    # print(sat)

    masterhdf = f'{sat}_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(masterhdf)

    with pd.HDFStore(masterhdfdir+masterhdf, 'r') as store:
        times = store.select_column('/mlt', 'index').values # the time index

    # align external data with satellite measurements
    sat_external = external.reindex(times, method = 'nearest', tolerance = '2Min')

    if doAddSubstorms:

        print("Adding substorm onset info ...")

        # for istorm in range(ss.shape[0]):
            # ssdt = np.diff(ss.index)
            # bins = np.arange(0,91,10)
            # np.histogram(ssdt,(bins*60*1e9).astype(np.timedelta64))
            # np.digitize(ssdt.astype(np.float64),(bins*60*1e9).astype(np.timedelta64).astype(np.float64))

            # The TimedeltaIndex way, which doesn't allow for use of np.digitize 
            # ssdt = pd.TimedeltaIndex(ssdt)
            # bins = pd.TimedeltaIndex(data=bins,unit='m')
            # np.histogram(ssdt,bins=bins)

        from hatch_python_utils.arrays import value_locate
        print(f"Aligning {times.shape[0]} timestamps with {ss.shape[0]} substorm onset times (patience) ...")
        whar = value_locate(ss.index.to_numpy(),times)
        onset_dt = times-ss.iloc[whar].index  # when measurement occurs relative to nearest substorm onset
        onset_mlat = ss.iloc[whar]['mlat'].values
        onset_mlt = ss.iloc[whar]['mlt'].values

        # if meas occurs outside tolerance window, make it nan
        print(f"Applying {ss_maxdt_tolerance.replace(' ','-')} tolerance window to substorm params")
        bads = np.abs(onset_dt) > pd.Timedelta(ss_maxdt_tolerance)  

        onset_dt = onset_dt.total_seconds().values/60  # Convert to float minutes so that pandas doesn't complain
        onset_dt[bads] = np.nan
        onset_mlat[bads] = np.nan
        onset_mlt[bads] = np.nan

        # Add what we learned to sat_external for loading into hdf file
        sat_external['onset_dt_minutes'] = onset_dt
        sat_external['onset_mlat'] = onset_mlat
        sat_external['onset_mlt'] = onset_mlt

    with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
        store.append('/external', sat_external, data_columns = True, append = False)
        print('added %s' % (sat + '/external'))
