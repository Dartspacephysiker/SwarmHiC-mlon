# import importlib
# import hatch_python_utils.satellites.Swarm_TII
# importlib.reload(hatch_python_utils.satellites.Swarm_TII)
from hatch_python_utils.satellites.Swarm_TII import calculate_satellite_frame_vectors_in_NEC_coordinates,calculate_crosstrack_flow_in_NEC_coordinates

datapath = '/SPENCEdata/Research/database/SHEIC/'
# storefn = '/SPENCEdata/Research/database/SHEIC/data_v1_update.h5'
# groups = ['SwarmA', 'SwarmB', 'SwarmC']

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A']
VERSION = '0302'
masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
hdfsuff = '_NOWDAT'

getcols = ['Bx','By','Bz',
           'Latitude','Longitude','Radius',
           'Viy',
           'mlat','mlon','mlt',
           'VsatN','VsatE','VsatC',
           'd10','d11','d12',
           'd20','d21','d22']

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


##############################
# Get convection in geodetic and Apex coordinates
# Bonus: get altitude and geodetic latitude
Viy_NEC = calculate_crosstrack_flow_in_NEC_coordinates(df)

df['Viy_N'] = Viy_NEC[0,:]
df['Viy_E'] = Viy_NEC[1,:]
df['Viy_C'] = Viy_NEC[2,:]

# Convert 'Viy_N' (and 'Viy_C') from geocentric to geodetic coordinates
gdlat, alt, Viy_NGeod, Viy_DGeod = geodesy.geoc2geod(90.-df["Latitude"].values,
                                                     df["Radius"].values/1000.,
                                                     -df['Viy_N'],df['Viy_C'])
                                     # -df.B_N, -df.B_C)

# Now Apex coords
# REMEMBER (from Richmond, 1995):
# • d1 base vector points "more or less in the magnetic eastward direction"
# • d2 base vector points "generally downward and/or equatorward (i.e., southward in NH [and in SH?])" 
df['Viy_d1'] = df['d10']*df['Viy_E'] + df['d11']*Viy_NGeod + df['d12']*Viy_DGeod
df['Viy_d2'] = df['d20']*df['Viy_E'] + df['d21']*Viy_NGeod + df['d22']*Viy_DGeod

df['gdlat'] = gdlat
df['alt'] = alt


##############################
# Get Weimer values

df['Bmag'] = np.sqrt((df[['Bx','By','Bz']]**2).sum(axis=1))




