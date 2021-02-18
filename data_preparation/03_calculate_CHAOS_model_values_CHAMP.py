import pandas as pd
import datetime as dt
import dask.array as da
import numpy as np
from chaosmagpy import load_CHAOS_matfile


FILEPATH_CHAOS = '/Users/laundal/Dropbox/CHAOS-6-x7.mat'
model = load_CHAOS_matfile(FILEPATH_CHAOS)
model_arr = lambda x1, x2, x3, x4: np.vstack(model(x1, x2, x3, x4)).T

CHUNKSIZE = 20000



names    = ['md2000', 'r', 'theta', 'phi', 'U_gc', 'B_theta', 'E_gc', 'B_r_mod_int', 'B_theta_mod_int', 'B_phi_mod_int', 'B_r_mod_ext', 'B_theta_mod_ext', 'B_phi_mod_ext', 'dB_Ugc_old', 'dB_theta_old', 'dB_Egc_old']

raw_df = pd.read_table('/Users/laundal/Dropbox/data/space/CHAMP/CHAMP_CSC_36_residuals_NEC_all.dat', skipinitialspace=True, skiprows = [0, 1], sep = ' ', names = names)
index = raw_df['md2000']*24*60**2 * dt.timedelta(seconds = 1) + dt.datetime(2000, 1, 1)
index = pd.DatetimeIndex(index).round('1S')

raw_df.index = index

dB_Egc_old =  raw_df['dB_Egc_old']
dB_Ugc_old =  raw_df['dB_Ugc_old']
dB_Ngc_old = -raw_df['dB_theta_old']

mjd2000_da = da.from_array(raw_df['md2000'].values , chunks = CHUNKSIZE)
colat_da   = da.from_array(raw_df['theta' ].values , chunks = CHUNKSIZE)
lon_da     = da.from_array(raw_df['phi'   ].values , chunks = CHUNKSIZE)
rkm_da     = da.from_array(raw_df['r'     ].values , chunks = CHUNKSIZE)


dd = da.map_blocks(model_arr, *(mjd2000_da, rkm_da, colat_da, lon_da), new_axis = 1, dtype = np.float64, chunks = (3, CHUNKSIZE))
ds = dd.compute()

Br, Btheta, Bphi = ds.T   

store = pd.HDFStore('CHAMP_v1_update.h5', 'a')

store.append('gclat', 90 - raw_df['theta'], format='t')
store.append('gclon',      raw_df['phi'], format='t')
store.append('rkm'  ,      raw_df['r'], format='t')

store.append('U_gc_old', dB_Egc_old, format='t')
store.append('N_gc_old', dB_Ugc_old, format='t')
store.append('E_gc_old', dB_Ngc_old, format='t')

store.append('U_gc_CHAOS', pd.Series( Br    , index = index), format='t')
store.append('N_gc_CHAOS', pd.Series(-Btheta, index = index), format='t')
store.append('E_gc_CHAOS', pd.Series( Bphi  , index = index), format='t')

store.append('E_gc',  raw_df['E_gc']   , format = 't')
store.append('N_gc', -raw_df['B_theta'], format = 't')
store.append('U_gc',  raw_df['U_gc']   , format = 't')




store.close()




