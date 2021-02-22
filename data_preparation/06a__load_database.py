import pandas as pd
import numpy as np

########################################
# Select what you want to load
# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A']
VERSION = '0302'
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
hdfsuff = '_NOWDAT'
hdfsuff = '_Anna'
hdfsuff = '_Anna2'
# hdfsuff = '_2014'


########################################
# Don't modify past here

ALLCOLS = ['Bx', 'By', 'Bz',              # Magnetic field measurements made by Swarm in the spacecraft frame of reference
           'Ehx', 'Ehy', 'Ehz',           # Electric field measurements 
           'Evx', 'Evy', 'Evz',
           'Latitude', 'Longitude', 'Radius', 
           'MLT', 'QDLatitude',           # QD coordinates
           'mlat', 'mlon', 'mlt',         # MA-110 coordinates
           'Quality_flags',               # Use Quality_flags >= 4
           'Vicrx', 'Vicry', 'Vicrz',
           'Vixh', 'Vixh_error',
           'Vixv', 'Vixv_error',
           'Viy', 'Viy_error',
           'Viy_d1', 'Viy_d2',
           'Viz', 'Viz_error', 
           'ViyWeimer_d1', 'ViyWeimer_d2',
           'VsatN', 'VsatE', 'VsatC',
           'd10', 'd11', 'd12',
           'd20', 'd21', 'd22',
           'd30', 'd31', 'd32',
           'external']          # 'external' contains IMF/SW variables in getexternals list


getcols = [#'Bx','By','Bz',
    'Quality_flags',
    'Latitude','Longitude','Radius',
    'Viy', 'Viy_d1', 'Viy_d2',
    'ViyWeimer_d1', 'ViyWeimer_d2',
    'mlat','mlon','mlt']#,
    # 'VsatN','VsatE','VsatC',
    # 'd10','d11','d12',
    # 'd20','d21','d22']
getexternals = ['Bz', 'By', 'vx', 'nsw', 'tilt', 'f107obs', 'f107adj']

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

        for wantextcol in getexternals:
            colname = 'IMF'+wantextcol if (wantextcol[0] == 'B') else wantextcol
            df[colname] = store['/external'][wantextcol]
