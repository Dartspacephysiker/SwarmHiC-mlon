""" script to read HDF store, calculate all the weights, and store in weights, coordinates, and measurements 
    in  a format which can be streamed (using dask)     
"""

import dask.array as da
import numpy as np
import pandas as pd
from pytt.utils.sh import nterms

NT, MT, NV, MV = 65, 3, 45, 3
NEQ = nterms(NT, MT, NV, MV)


hdfstorage = '/Users/laundal/Dropbox/data/space/LEO/data_v1_update.h5' # where the data is stored
output = '/Users/laundal/Dropbox/science/projects/poloidal_toroidal/model_v1/data/modeldata_v1_update.hdf5' # where the data will be stored

satellites = ['CHAMP', 'SwarmA', 'SwarmB', 'SwarmC']

columns = ['h', 'qdlat', 'alat110', 'mlt', 'Be', 'Bn', 'Bu', 'f1e', 'f1n', 'f2e', 'f2n', 'd1e', 'd1n', 'd2e', 'd2n']
ext_columns = ['vx', 'Bz', 'By', 'f107', 'tilt']


# put the satellite data in a dict of dataframes:
subsets = {}
external = {}
with pd.HDFStore(hdfstorage, 'r') as store:
    for satellite in satellites:
        print ('reading %s' % satellite)
        subsets[satellite]  = store.select(satellite + '/apex_data', columns = columns)
        external[satellite] = store.select(satellite + '/external', columns = ext_columns)

# add alpha charlie-weight to main df:
print ('adding satellite weights')
for satellite in satellites:
    if satellite in ['SwarmA', 'SwarmC']:
        subsets[satellite]['s_weight'] = 0.5
    else:
        subsets[satellite]['s_weight'] = 1.0

# add external parameters to main df - and drop nans:
print ('adding external to main dataframe')
for satellite in satellites:
    subsets[satellite][external[satellite].columns] = external[satellite]
    length = len(subsets[satellite])
    subsets[satellite] = subsets[satellite].dropna()
    print ('dropped %s out of %s datapoints because of nans' % (length - len(subsets[satellite]), length))

print ('merging the subsets')
full = pd.concat(subsets)
full['time'] = full.index
full.index = range(len(full))



# calculate weights:
print ('calculating weights')
B = np.sqrt(full['By']**2 + full['Bz']**2)
ca = np.arctan2(full['By'], full['Bz'])
epsilon = full['vx'].abs()**(4/3.) * B**(2/3.) * (np.sin(ca/2)**(8))**(1/3.) / 1000 # mV/m # Newell coupling
tau     = full['vx'].abs()**(4/3.) * B**(2/3.) * (np.cos(ca/2)**(8))**(1/3.) / 1000 # mV/m # Newell coupling
f107    = full['f107']
tilt = full['tilt']
full['w01' ] = 1              * np.sin(ca)
full['w02' ] = 1              * np.cos(ca)
full['w03' ] = epsilon
full['w04' ] = epsilon        * np.sin(ca)
full['w05' ] = epsilon        * np.cos(ca)
full['w06' ] = tilt
full['w07' ] = tilt           * np.sin(ca)
full['w08' ] = tilt           * np.cos(ca)
full['w09' ] = tilt * epsilon
full['w10' ] = tilt * epsilon * np.sin(ca)
full['w11' ] = tilt * epsilon * np.cos(ca)
full['w12' ] = tau
full['w13' ] = tau            * np.sin(ca)
full['w14' ] = tau            * np.cos(ca)
full['w15' ] = tilt * tau    
full['w16' ] = tilt * tau     * np.sin(ca)
full['w17' ] = tilt * tau     * np.cos(ca)
full['w18' ] = f107.copy()



columns = ['h', 'qdlat', 'alat110', 'mlt', 'Be', 'Bn', 'Bu', 'f1e', 'f1n', 'f2e', 'f2n', 'd1e', 'd1n', 'd2e', 'd2n', 's_weight', 
           'w01', 'w02', 'w03', 'w04', 'w05', 'w06', 'w07', 'w08', 'w09', 'w10', 'w11', 'w12', 'w13', 'w14', 'w15', 'w16', 'w17', 'w18', 
           'time', 'sat_identifier']
sat  = [k[0] for k in full['time']]
time = [k[1] for k in full['time']]
full['time'] = time
full['sat'] = sat
full['sat_identifier'] = full['sat'].map({'CHAMP':1, 'SwarmA':2, 'SwarmB':3, 'SwarmC':4})
full['time'] = np.float64(full['time'].values)

print (full.shape)
chunksize = NEQ

newstore = da.to_hdf5(output, dict(zip(['data/' + k for k in columns], [da.from_array(full[k].values, chunksize) for k in columns])))
print ('put model data in %s' % output)
