import h5py
import pandas as pd
import numpy as np
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
# DATAVERSION = 'v1'
DATAVERSION = 'v2'                                       # 2021/11/19
inputfile       = f'modeldata_{DATAVERSION}_update.hdf5' # where the data are stored (see data_preparation/07_make_model_dataset.py)
datafile        = masterhdfdir+inputfile

##############################
# Quality flag enforcement
##############################
# Do you want to only include measurements with a certain quality flag?
# For data version 0302, the second bit ('0100') being set means that v_(i,y) is calibrated
# See section 3.4.1.1 in "EFI TII Cross-Track Flow Data Release Notes" (Doc. no: SW-RN-UOC-GS-004, Rev: 7)
enforce_quality_flag = True     
quality_flag = '0100'

# For example, as of 20211119 if you do the following with
#store = pd.HDFStore('Sat_A_ct2hz_v0302_5sres.h5','r'),
#you get '9047089' back:
#np.sum((store['/Quality_flags'].values & int(quality_flag,2)) > 0)
# Not sure if we need these
# prefix_GTd_GTG_fn    = masterhdfdir+'matrices/GTG_GTd_array_iteration_'
# prefix_model_fn      = masterhdfdir+'matrices/model_v1_iteration_'
# prefix_model_value   = masterhdfdir+'matrices/model_v1_values_iteration_'
# prefix_huber_weights = masterhdfdir+'matrices/model_v1_huber_iteration_'

f = h5py.File(datafile, 'r')['/data']
names = [item[0] for item in f.items()]
datamap = dict(zip(names, range(len(names))))


VERSION = '0302'
hdfsuff = '_5sres'

sats = ['Sat_A','Sat_B']
satmap = {'Sat_A':1, 'Sat_B':2}
# stores = {}
# dfs = {}

columns = ['mlat', 'mlt','lperptoB_dot_e1','lperptoB_dot_e2']
choosef107 = 'f107obs'
ext_columns = ['vx', 'Bz', 'By', choosef107, 'tilt']
# Derived columns: ["lperptoB_dot_ViyperptoB","Be3_in_Tesla","D"]

subsets = {}
external = {}
subinds = {}

# indies = slice(0,None)
# indies = slice(0,2000000)
# indies = slice(3000000,6000000)

# Sub index for making model
do_getsubinds = True
# outindfile = 'negbz_array_indices.txt'
outindfile = 'negby_array_indices.txt'
outindfile = 'posby_array_indices.txt'
outindfile = 'sortiment_array_indices.txt'
outindfile = 'alldptilt_array_indices.txt'

# if do_getsubinds:
#     if outindfile == 'negbz_array_indices.txt':
#         indfunc = lambda full: np.where((full['mlat'] > 0 ) & \
#                                         (full['Bz'] <= 0) & \
#                                         (np.abs(full['tilt']) <= 10) & \
#                                         ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

#         # ~4% of entire database, 646620 indices
#         #indlets.size/len(full)
#         #Out[24]: 0.038931557476638776
#     elif outindfile == 'negby_array_indices.txt':
#         indfunc = lambda full: np.where((full['mlat'].abs() >= 45 ) & \
#                                         (full['By'] <= 0) & \
#                                         (np.abs(full['tilt']) <= 10) & \
#                                         ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

#     elif outindfile == 'posby_array_indices.txt':
#         indfunc = lambda full: np.where((full['mlat'].abs() >= 45 ) & \
#                                         (full['By'] >= 0) & \
#                                         (np.abs(full['tilt']) <= 10) & \
#                                         ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

#     elif outindfile == 'sortiment_array_indices.txt':
#         indfunc = lambda full: np.where((full['mlat'].abs() >= 45 ) & \
#                                         # (full['By'] >= 0) & \
#                                         (np.abs(full['tilt']) <= 10) & \
#                                         ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

#     else:
#         assert 2<0

for sat in sats:
    inputhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'
    print("Opening "+inputhdf)
    # stores[sat.replace('Sat_','')] = pd.HDFStore(masterhdfdir+inputhdf, 'r')
    with pd.HDFStore(masterhdfdir+inputhdf, 'r') as store:

        # uses indies
        # print("Getting main columns ... ",end='')
        # tmpdf = pd.DataFrame()
        # for wantcol in columns:
        #     tmpdf[wantcol] = store['/'+wantcol].iloc[indies]
        
        # print("Getting ext columns ...",end='')
        # tmpextdf = store.select('/external', columns = ext_columns).iloc[indies]

        # print("Getting derived quantities ...")
        # print("D",end=', ')
        # D = np.sqrt( (store['d11']*store['d22']-store['d12']*store['d21'])**2 + \
        #              (store['d12']*store['d20']-store['d10']*store['d22'])**2 + \
        #              (store['d10']*store['d21']-store['d11']*store['d20'])**2).iloc[indies]
        # tmpdf['D'] = D.values
        # print("Be3_in_Tesla",end=', ')
        # tmpdf['Be3_in_Tesla'] = store['B0IGRF'].iloc[indies]/D.values/1e9
        # print("lperptoB_dot_ViyperptoB",end=', ')
        # tmpdf['lperptoB_dot_ViyperptoB'] = store['/lperptoB_E'].iloc[indies]*store['/ViyperptoB_E'].iloc[indies]+\
        #     store['/lperptoB_N'].iloc[indies]*store['/ViyperptoB_N'].iloc[indies]+\
        #     store['/lperptoB_U'].iloc[indies]*store['/ViyperptoB_U'].iloc[indies]

        # DOESN'T use indies
        print("Getting main columns ... ",end='')
        tmpdf = pd.DataFrame()
        for wantcol in columns:
            tmpdf[wantcol] = store['/'+wantcol]
        
        print("Getting ext columns ...",end='')
        tmpextdf = store.select('/external', columns = ext_columns)

        print("Getting derived quantities ...")
        print("D",end=', ')
        D = np.sqrt( (store['d11']*store['d22']-store['d12']*store['d21'])**2 + \
                     (store['d12']*store['d20']-store['d10']*store['d22'])**2 + \
                     (store['d10']*store['d21']-store['d11']*store['d20'])**2)
        tmpdf['D'] = D.values
        print("Be3_in_Tesla",end=', ')
        tmpdf['Be3_in_Tesla'] = store['B0IGRF']/D.values/1e9
        print("lperptoB_dot_ViyperptoB",end=', ')
        tmpdf['lperptoB_dot_ViyperptoB'] = store['/lperptoB_E']*store['/ViyperptoB_E']+\
            store['/lperptoB_N']*store['/ViyperptoB_N']+\
            store['/lperptoB_U']*store['/ViyperptoB_U']


        if enforce_quality_flag:
        
            print(f"Dropping records that do not have Quality_flag == {quality_flag} ...")
        
            nNow = len(tmpdf)
            tmpdf = tmpdf[(store['/Quality_flags'].values & int(quality_flag,2)) > 0]
            nLater = len(tmpdf)
                
            print(f"Dropped {nNow-nLater} of {nNow} records ({(nNow-nLater)/nNow*100}%)")

    subsets[sat] = tmpdf
    external[sat] = tmpextdf
    
    # if do_getsubinds:
    #     subinds[sat] = indfunc

print ('adding satellite weights ...',end='')
for sat in sats:
    # if sat in ['SwarmA', 'SwarmC']:
    #     subsets[sat]['s_weight'] = 1.0
    # else:
    subsets[sat]['s_weight'] = 1.0

# add external parameters to main df - and drop nans:
print ('adding external to main dataframe')
for sat in sats:
    subsets[sat][external[sat].columns] = external[sat]
    length = len(subsets[sat])
    subsets[sat] = subsets[sat].dropna()
    # subsets[sat] = subsets[sat][subsets[sat][choosef107] <= 350]
    print ('dropped %s out of %s datapoints because of nans' % (length - len(subsets[sat]), length))

print ('merging the subsets')
full = pd.concat(subsets)
full['time'] = full.index
full.index = range(len(full))

sat  = [k[0] for k in full['time']]
time = [k[1] for k in full['time']]
full['time'] = time
full['sat'] = sat
full['sat_identifier'] = full['sat'].map(satmap)
full['time'] = np.float64(full['time'].values)

# Sub index for making model
breakpoint()
if do_getsubinds:
    if outindfile == 'negbz_array_indices.txt':
        indlets = np.where((full['mlat'] > 0 ) & \
                           (full['Bz'] <= 0) & \
                           (np.abs(full['tilt']) <= 10) & \
                           ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]
        # ~4% of entire database, 646620 indices
        #indlets.size/len(full)
        #Out[24]: 0.038931557476638776
    elif outindfile == 'negby_array_indices.txt':
        indlets = np.where((full['mlat'].abs() >= 45 ) & \
                           (full['By'] <= 0) & \
                           (np.abs(full['tilt']) <= 10) & \
                           ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

    elif outindfile == 'posby_array_indices.txt':
        indlets = np.where((full['mlat'].abs() >= 45 ) & \
                           (full['By'] >= 0) & \
                           (np.abs(full['tilt']) <= 10) & \
                           ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

    elif outindfile == 'sortiment_array_indices.txt':
        indlets = np.where((full['mlat'].abs() >= 45 ) & \
                           # (full['By'] >= 0) & \
                           (np.abs(full['tilt']) <= 10) & \
                           ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

    elif outindfile == 'alldptilt_array_indices.txt':
        indlets = np.where((full['mlat'].abs() >= 45 ) & \
                           # (full['By'] >= 0) & \
                           # (np.abs(full['tilt']) <= 10) & \
                           ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]

    else:
        assert 2<0

    # if indies.start > 0:
    #     print(f"Adding indies.start (= {indies.start}) to indlets")
    #     indlets += indies.start

    print(f"Saving {len(indlets)} indices to outindfile '{outindfile}'")
    np.savetxt(masterhdfdir+outindfile,indlets,fmt='%d')


# First verify that we got the right guys
indies2 = slice(0,1000000,100)

# f['time'][()][indies2]-full.iloc[indies2]['time'].values
print("Where mismatches:")
misematchem = np.where(np.abs(f['time'][()][indies2]-full.iloc[indies2]['time'].values) > 0)[0]
print(misematchem)
if len(misematchem) == 0:
    print("Good!")
else:
    assert 2<0,"Åneiånei!"

mini = full.iloc[indies2]

fig,ax = plt.subplots(1,1)
ax.scatter(mini['mlat'],mini['mlt'],alpha=0.05)

fig,ax = plt.subplots(1,1)
ax.scatter(mini['f107obs'],mini['tilt'],alpha=0.05)
ax.set_xlabel('F10.7')
ax.set_xlabel('Tilt [deg]')

fig,ax = plt.subplots(1,1)
ax.scatter(mini['Bz'],mini['By'],alpha=0.05)
ax.set_xlabel('Bz [nT]')
ax.set_ylabel('By [nT]')

fig,ax = plt.subplots(1,1)
ax.scatter(mini['Bz'],mini['vx'],alpha=0.05)
ax.set_xlabel('Bz [nT]')
ax.set_ylabel('vx [km/s]')

#To compare with f:
np.float64(store['mlat'].iloc[indies].index.values)
