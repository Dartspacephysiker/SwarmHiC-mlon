import pandas as pd
import datetime as dt
import dask.array as da
import numpy as np
# from chaosmagpy import load_CHAOS_matfile
from chaosmagpy import CHAOS
import chaosmagpy as cp
import os

print("There's not a huge amount of difference between the magnitude of the Swarm-measured mag field and the CHAOS field magnitude. Does a nT (i.e., one part in twenty thousand) or two really matter?")
print("That is, should we just use the Swarm-measured field and not waste our time?")

CHAOSDIR = '/home/spencerh/OneDrive/Research/database/Chaos/'
CHAOSFIL = 'CHAOS-6-x7.mat'
# CHAOSFIL = 'CHAOS-7.mat'
CHAOSFIL = 'CHAOS-7.8.mat'
FILEPATH_CHAOS = CHAOSDIR+CHAOSFIL

# model = load_CHAOS_matfile(FILEPATH_CHAOS)
model = cp.CHAOS.from_mat(FILEPATH_CHAOS)
model_arr = lambda x1, x2, x3, x4: np.vstack(model(x1, x2/1000, x3, x4)).T  # Divide by 1000 to get radius in km

# CHUNKSIZE = 20000 # Works fine on Kalle's tower, but I assume too big for my laptop? Haven't tried ...
CHUNKSIZE = 5000 # Maybe?

masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
VERSION = '0302'

mode = 'Annadb2'
mode = '2014'
mode = 'fulldb_5sres'
# mode = 'fulldb_1sres'

VALIDMODES = ['fulldb_5sres','fulldb_1sres','Annadb','Annadb2','2014']

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A','Sat_B']
# sats = ['Sat_B','Sat_C']

assert mode in VALIDMODES,"Must choose one of " + ",".join(VALIDMODES)+"!"

if mode == 'fulldb_5sres':
    hdfsuff = '_5sres'

    decimationfactor = 10           # so 5-s resolution
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

if mode == 'fulldb_1sres':
    hdfsuff = '_1sres'

    decimationfactor = 2           # so 1-s resolution
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2021-01-01 00:00:00'

elif mode == 'Annadb':
    hdfsuff = '_Anna'

    decimationfactor = 1           # so full resolution (2 Hz)
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2014-01-01 00:00:00'

elif mode == 'Annadb2':
    hdfsuff = '_Anna2'

    decimationfactor = 1           # so full resolution (2 Hz)
    mlatlowlim = 45

    date0 = '2016-01-25 00:00:00'
    date1 = '2016-02-01 00:00:00'

elif mode == '2014':
    hdfsuff = '_2014'

    decimationfactor = 1           # so full resolution (2 Hz)
    mlatlowlim = 45

    date0 = '2013-12-01 00:00:00'
    date1 = '2014-12-31 23:59:59'


# store_files = ['SW_OPER_MAGA_LR_1B_raw.h5', 'SW_OPER_MAGB_LR_1B_raw.h5', 'SW_OPER_MAGC_LR_1B_raw.h5']

assert os.path.exists(masterhdfdir),"Doesn't exist: "+masterhdfdir

for sat in sats:

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    print("Doing "+masterhdf)

    fn = masterhdfdir+masterhdf

    store = pd.HDFStore(fn, 'a')

    inds = slice(None,None)  # All elements
    inds = slice(0,10)

    index = store['/Radius'].index[inds]
    mjd2000 = (pd.DatetimeIndex(index) - dt.datetime(2000, 1, 1) ).total_seconds() / (24.*60.**2) 
    # mjd2000 = cp.data_utils.mjd2000(index.year,index.month,index.day,index.hour,index.minute,index.second,index.microsecond)
    mjd2000_da = da.from_array(mjd2000, chunks = CHUNKSIZE)
    colat_da   = da.from_array(90 - store['/Latitude'][inds].values , chunks = CHUNKSIZE)
    lon_da     = da.from_array(     store['/Longitude'][inds].values, chunks = CHUNKSIZE)
    r_da       = da.from_array(     store['/Radius'][inds].values   , chunks = CHUNKSIZE)  # This provides radius in meters. We divide by 1000 to get radius in km in definition of model_arr above

    dd = da.map_blocks(model_arr, *(mjd2000_da, r_da, colat_da, lon_da), new_axis = 1, dtype = np.float64, chunks = (3, CHUNKSIZE))
    ds = dd.compute()

    Br, Btheta, Bphi = ds.T   

    B0 = np.sqrt(Br**2+Btheta**2+Bphi**2)

    B0Swarm = np.sqrt(store['/Bx'][inds]**2+store['/By'][inds]**2+store['/Bz'][inds]**2)
    D = np.sqrt( (store['/d11'][inds]*store['/d22'][inds]-store['/d12'][inds]*store['/d21'][inds])**2 + \
                 (store['/d12'][inds]*store['/d20'][inds]-store['/d10'][inds]*store['/d22'][inds])**2 + \
                 (store['/d10'][inds]*store['/d21'][inds]-store['/d11'][inds]*store['/d20'][inds])**2)

    Be3 = B0Swarm/D

    print("There's not a huge amount of difference between the magnitude of the Swarm-measured mag field and the CHAOS field magnitude. Does a nT (i.e., one part in twenty thousand) or two really matter?")
    print("That is, should we just use the Swarm-measured field and not waste our time?")

    breakpoint()

    # store.append('U_gc_CHAOS', pd.Series( Br    , index = index), format='t')
    # store.append('N_gc_CHAOS', pd.Series(-Btheta, index = index), format='t')
    # store.append('E_gc_CHAOS', pd.Series( Bphi  , index = index), format='t')

    store.append('B0_CHAOS', pd.Series( B0    , index = index), format='t')

    store.close()

