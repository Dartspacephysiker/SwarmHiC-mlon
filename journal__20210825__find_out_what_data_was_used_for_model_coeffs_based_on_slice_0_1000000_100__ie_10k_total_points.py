import h5py
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
output       = 'modeldata_v1_update.hdf5' # where the data will be stored

datafile             = masterhdfdir+'modeldata_v1_update.hdf5'
prefix_GTd_GTG_fn    = masterhdfdir+'matrices/GTG_GTd_array_iteration_'
prefix_model_fn      = masterhdfdir+'matrices/model_v1_iteration_'
prefix_model_value   = masterhdfdir+'matrices/model_v1_values_iteration_'
prefix_huber_weights = masterhdfdir+'matrices/model_v1_huber_iteration_'
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
subsets = {}
external = {}
indies = slice(0,2000000)
indies = slice(0,None)
for sat in sats:
    inputhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'
    print("Opening "+inputhdf)
    # stores[sat.replace('Sat_','')] = pd.HDFStore(masterhdfdir+inputhdf, 'r')
    with pd.HDFStore(masterhdfdir+inputhdf, 'r') as store:

        print("Getting main columns ... ",end='')
        tmpdf = pd.DataFrame()
        for wantcol in columns:
            tmpdf[wantcol] = store['/'+wantcol].iloc[indies]
        
        print("Getting ext columns ...",end='')
        tmpextdf = store.select('/external', columns = ext_columns).iloc[indies]

    subsets[sat] = tmpdf
    external[sat] = tmpextdf
    
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
do_getsubinds = True
if do_getsubinds:
    indlets = np.where((full['mlat'] > 0 ) & \
                       (full['Bz'] <= 0) & \
                       (np.abs(full['tilt']) <= 10) & \
                       ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]
    # ~4% of entire database, 646620 indices
    #indlets.size/len(full)
    #Out[24]: 0.038931557476638776

    outindfile = 'negbz_array_indices.txt'
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
