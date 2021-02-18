import pandas as pd
import numpy as np
from dipole import dipole_tilt

datapath = '/Users/laundal/Dropbox/data/space/'
storefn = '/Users/laundal/Dropbox/data/space/LEO/data_v1_update.h5'
groups = ['CHAMP', 'SwarmA', 'SwarmB', 'SwarmC']

y1 = '2000'
y2 = '2018'

PERIOD = '20Min'


f107 = pd.read_csv(datapath + 'noaa_radio_flux.csv', sep = ',', parse_dates= True, index_col = 0)  


# load omni

with pd.HDFStore(datapath + 'omni.h5', 'r') as omni:
    Bz = omni['Bz_nT_GSM'][y1:y2].rolling(PERIOD).mean()
    By = omni['By_nT_GSM'][y1:y2].rolling(PERIOD).mean()
    vx = omni['Vx_Velocity_km_per_s_GSE'][y1:y2].rolling(PERIOD).mean()

external = pd.DataFrame([Bz, By, vx]).T
external.columns = ['Bz', 'By', 'vx']
external = external.dropna()

# calcualte tilt
external['tilt'] = np.nan
for year in np.unique(external.index.year):
    print('calculting tilt for %s' % year)
    external.loc[str(year), 'tilt'] = dipole_tilt(external[str(year)].index, year)

# interpolate f107: 
f107[f107 < 0] = np.nan
# there is a huge data gap last 8 months of 2018 - I just inteprolate over this
f107 = f107.reindex(external.index).interpolate(method = 'linear', limit = 24*60*8*31)
external['f107'] = f107


for group in groups:
    print(group)
    with pd.HDFStore(storefn, 'r') as store:
        times = store.select_column(group + '/apex_data', 'index').values # the time index

    # align external data with satellite measurements
    sat_external = external.reindex(times, method = 'nearest', tolerance = '2Min')

    with pd.HDFStore(storefn, 'a') as store:
        store.append(group + '/external', sat_external, data_columns = True, append = False)
        print('added %s' % (group + '/external'))
