import pandas as pd
import datetime as dt
import dask.array as da
import numpy as np
from chaosmagpy import load_CHAOS_matfile


FILEPATH_CHAOS = '/Users/laundal/Dropbox/CHAOS-6-x7.mat'
model = load_CHAOS_matfile(FILEPATH_CHAOS)
model_arr = lambda x1, x2, x3, x4: np.vstack(model(x1, x2, x3, x4)).T


CHUNKSIZE = 20000

store_files = ['SW_OPER_MAGA_LR_1B_raw.h5', 'SW_OPER_MAGB_LR_1B_raw.h5', 'SW_OPER_MAGC_LR_1B_raw.h5']

for fn in store_files:
    store = pd.HDFStore(fn, 'a')

    index = store['/E_gc'].index
    mjd2000 = (pd.DatetimeIndex(index) - dt.datetime(2000, 1, 1) ).total_seconds() / (24.*60.**2) 
    mjd2000_da = da.from_array(mjd2000, chunks = CHUNKSIZE)
    colat_da = da.from_array(90 - store['/gclat'].values, chunks = CHUNKSIZE)
    lon_da = da.from_array(       store['/gclon'].values  , chunks = CHUNKSIZE)
    rkm_da = da.from_array(       store['/r_km'].values   , chunks = CHUNKSIZE)

    
    dd = da.map_blocks(model_arr, *(mjd2000_da, rkm_da, colat_da, lon_da), new_axis = 1, dtype = np.float64, chunks = (3, CHUNKSIZE))
    ds = dd.compute()

    Br, Btheta, Bphi = ds.T   

    store.append('U_gc_CHAOS', pd.Series( Br    , index = index), format='t')
    store.append('N_gc_CHAOS', pd.Series(-Btheta, index = index), format='t')
    store.append('E_gc_CHAOS', pd.Series( Bphi  , index = index), format='t')

    store.close()



