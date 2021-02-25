import pandas as pd
import numpy as np
from dipole import dipole_tilt
from datetime import datetime

datapath = '/SPENCEdata/Research/database/SHEIC/'
# storefn = '/SPENCEdata/Research/database/SHEIC/data_v1_update.h5'
# groups = ['SwarmA', 'SwarmB', 'SwarmC']

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_C']
VERSION = '0302'
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
hdfsuff = '_5sres'
# hdfsuff = '_Anna'
# hdfsuff = '_Anna2'
# hdfsuff = '_2014'


if hdfsuff == '_5sres':
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

##############################
# OMNI

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
f107[f107cols['f107obs']][f107[f107cols['f107obs']] < 0] = np.nan
f107[f107cols['f107adj']][f107[f107cols['f107adj']] < 0] = np.nan
# there is a huge data gap last 8 months of 2018 - I just inteprolate over this
f107 = f107.reindex(f107.index.union(external.index)).interpolate(method = 'linear', limit = 24*60*8*31)
for key,val in f107cols.items():
    external[key] = f107[val]

##############################
# Dipole tilt

external['tilt'] = np.nan
for year in np.unique(external.index.year):
    print('calculting tilt for %s' % year)
    external.loc[str(year), 'tilt'] = dipole_tilt(external[str(year)].index, year)

##############################
# Substorms

# ssinfile = '/home/spencerh/Desktop/substorms-ohtani-20131201_000000_to_20211202_000000.ascii'
ssinfile = '/SPENCEdata/Research/database/SHEIC/substorms-ohtani-20131201_000000_to_20211202_000000.ascii'

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

breakpoint()

##############################
# Add data to master hdfs

# ORIG
# for group in groups:
#     print(group)
#     with pd.HDFStore(storefn, 'r') as store:
#         times = store.select_column(group + '/apex_data', 'index').values # the time index

#     # align external data with satellite measurements
#     sat_external = external.reindex(times, method = 'nearest', tolerance = '2Min')

#     with pd.HDFStore(storefn, 'a') as store:
#         store.append(group + '/external', sat_external, data_columns = True, append = False)
#         print('added %s' % (group + '/external'))

# NY
for sat in sats:
    print(sat)

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(masterhdf)

    with pd.HDFStore(masterhdfdir+masterhdf, 'r') as store:
        times = store.select_column('/mlt', 'index').values # the time index

    # align external data with satellite measurements
    sat_external = external.reindex(times, method = 'nearest', tolerance = '2Min')

    with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
        store.append('/external', sat_external, data_columns = True, append = False)
        print('added %s' % (sat + '/external'))
