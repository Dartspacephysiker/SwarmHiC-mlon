from pysymmetry import geodesy
import pandas as pd
import numpy as np
from hatch_python_utils.math.vectors import dotprod
from hatch_python_utils.time_tools import doy_single    
from DAG.src.pysymmetry.models.weimer.convection import Convection
from hatch_python_utils.satellites.Swarm_TII import calculate_satellite_frame_vectors_in_NEC_coordinates,calculate_crosstrack_flow_in_NEC_coordinates

doAddWeimer = False             # Add Weimer model values? Takes a very long time …
Weimer_update_dt = '180 s'  # How often should we update the Weimer model?
maxtdiff_between_wanttime_and_omni = pd.Timedelta('2 min')

# sats = ['Sat_A','Sat_B','Sat_C']
sats = ['Sat_A']
# sats = ['Sat_B','Sat_C']
VERSION = '0302'
hdfsuff = '_5sres'
# hdfsuff = '_Anna'
# hdfsuff = '_Anna2'
# hdfsuff = '_2014'

masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'
datapath = '/SPENCEdata/Research/database/SHEIC/'

OMNIPERIOD = '20Min'
y1OMNI = '2013'
y2OMNI = '2021'

getcols = ['Bx','By','Bz',
           'Quality_flags',
           'Latitude','Longitude','Radius',
           'Viy',
           'mlat','mlon','mlt',
           'VsatN','VsatE','VsatC',
           'd10','d11','d12',
           'd20','d21','d22']


##############################
# Load OMNI (if we are planning to add Weimer convection)
if doAddWeimer:

    print("Loading OMNI database for calculating Weimer convection ...")
    omnicols = dict(bz = 'BZ_GSM',
                    by = 'BY_GSM',
                    vx = 'Vx',
                    nsw='proton_density')
    with pd.HDFStore(datapath + 'omni_1min.h5', 'r') as omni:
        external = omni['/omni']

    external = external[~external.index.duplicated(keep='first')]

    Bz = external[omnicols['bz']][y1OMNI:y2OMNI].rolling(OMNIPERIOD).mean()
    By = external[omnicols['by']][y1OMNI:y2OMNI].rolling(OMNIPERIOD).mean()
    vx = external[omnicols['vx']][y1OMNI:y2OMNI].rolling(OMNIPERIOD).mean()
    nsw = external[omnicols['nsw']][y1OMNI:y2OMNI].rolling(OMNIPERIOD).mean()

    external = pd.DataFrame([Bz, By, vx, nsw]).T
    external.columns = ['Bz', 'By', 'vx', 'nsw']
    external = external.dropna()

    def get_Weimer_convec_in_crosstrack_direction_in_apex_coords(df,vNWeimer,vEWeimer):
        """
        Name says it all!
        """
        # Satellite frame unit vectors in NEC coords
        xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=['VsatN','VsatE','VsatC'])
        
        # Satellite-frame y unit vector in Apex coords
        yhat_d1 = df['d10']*yhat[1,:] + df['d11']*yhat[0,:] + df['d12']*yhat[2,:]
        yhat_d2 = df['d20']*yhat[1,:] + df['d21']*yhat[0,:] + df['d22']*yhat[2,:]
        yhat_apex = np.array([yhat_d2*(-1),yhat_d1,yhat_d1*0])
        
        # Now dot Weimer vector (which is in AACGM/Apex coords) with yhat_apex vector to get magnitude of Weimer flow in cross-track direction
        vWeimer_yhat = dotprod(yhat_apex.T,np.array([vNWeimer,vEWeimer,vEWeimer*0]).T)
        
        return vWeimer_yhat

########################################    
# Loop over satellites

for sat in sats:
    # print(sat)

    masterhdf = sat+f'_ct2hz_v{VERSION}{hdfsuff}.h5'

    print(masterhdf)

    # Make df with getcols
    df = pd.DataFrame()

    with pd.HDFStore(masterhdfdir+masterhdf, 'r') as store:
        times = store.select_column('/mlt', 'index').values # the time index

        for wantcol in getcols:
            df[wantcol] = store['/'+wantcol]

    ##############################
    # Get convection in geodetic and Apex coordinates
    # Bonus: get altitude and geodetic latitude
    print("Getting crosstrack vector in NEC coordinates ...")
    Viy_NEC = calculate_crosstrack_flow_in_NEC_coordinates(df)
    
    df['Viy_N'] = Viy_NEC[0,:]
    df['Viy_E'] = Viy_NEC[1,:]
    df['Viy_C'] = Viy_NEC[2,:]
    
    # Convert 'Viy_N' (and 'Viy_C') from geocentric to geodetic coordinates
    print("Converting to geodetic coordinates ...")
    gdlat, alt, Viy_NGeod, Viy_DGeod = geodesy.geoc2geod(90.-df["Latitude"].values,
                                                         df["Radius"].values/1000.,
                                                         -df['Viy_N'],df['Viy_C'])
                                         # -df.B_N, -df.B_C)
    
    # Now Apex coords
    # REMEMBER (from Richmond, 1995):
    # • d1 base vector points "more or less in the magnetic eastward direction"
    # • d2 base vector points "generally downward and/or equatorward (i.e., southward in NH [and in SH?])" 
    print("Calculating Viy_d1, Viy_d2, yhat_d1, yhat_d2 ...")
    df['Viy_d1'] = df['d10']*df['Viy_E'] + df['d11']*Viy_NGeod + df['d12']*Viy_DGeod
    df['Viy_d2'] = df['d20']*df['Viy_E'] + df['d21']*Viy_NGeod + df['d22']*Viy_DGeod
    
    # Also get satellite yhat vector in Apex coordinates
    xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=['VsatN','VsatE','VsatC'])
    yhat_d1 = df['d10']*yhat[1,:] + df['d11']*yhat[0,:] + df['d12']*yhat[2,:]
    yhat_d2 = df['d20']*yhat[1,:] + df['d21']*yhat[0,:] + df['d22']*yhat[2,:]
    
    df['gdlat'] = gdlat
    df['alt'] = alt
    
    df['yhat_d1'] = yhat_d1
    df['yhat_d2'] = yhat_d2
    
    with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
        storecols = ['Viy_d1','Viy_d2', 'yhat_d1', 'yhat_d2', 'gdlat', 'alt']
        print(f"Storing {', '.join(storecols)} for {sat} ...")
        for column in storecols:
            store.append(column, df[column], format='t', append=False)
    
    ##############################
    # Get Weimer values
    
    if doAddWeimer:
    
        print("Now getting Weimer convection for {df.shape[0]} measurements ...")

        df['Bmag'] = np.sqrt((df[['Bx','By','Bz']]**2).sum(axis=1))
        
        # Loop over stuff with resolution 30 s
        td = pd.Timedelta(Weimer_update_dt)
        tmin = df.index.min()
        tmax = df.index.max()
        times = pd.date_range(tmin,tmax,freq=Weimer_update_dt) 
        
        df['ViyWeimer_d1'] = np.nan
        df['ViyWeimer_d2'] = np.nan
        
        for it,t in enumerate(times):
            print(f"{it:4d} {t.strftime('%Y-%m-%d %H:%M:%S')}",end=' ')
            hereinds = (df.index >= t) & (df.index < (t + td))
            nhere = hereinds.sum()
            if nhere == 0:
                print("Skip!")
                continue
            print(nhere)
        
        
            # Get Weimer params for this moment
        
            nearestind = np.abs(external.index-t).argsort()[0]
            if np.abs(external.iloc[nearestind].name-t) > maxtdiff_between_wanttime_and_omni:
                continue
        
            iyear = t.year
            iday = np.int64(np.floor(doy_single(t)))
            uthr = t.hour + (t.minute/60. + t.second/3600.)/24.
        
            Bz, By, vsw, nsw = external.iloc[nearestind].values
            vsw = np.abs(vsw)           # Weimer model wants it to be abs
            
            print(f"{iyear} {iday} {uthr:6.2f} {By:6.2f} {Bz:6.2f} {vsw:8.2f} {nsw:7.2f}")
        
            vNWeimer, vEWeimer, vBWeimer = Convection(df[hereinds]['mlat'].values,df[hereinds]['mlt'].values,
                                                      df[hereinds]['Bmag'].values,
                                                      iyear,iday,uthr,By,Bz,vsw,nsw)
            
            vWeimer_yhat = get_Weimer_convec_in_crosstrack_direction_in_apex_coords(df[hereinds],vNWeimer,vEWeimer)
        
            df.loc[hereinds,'ViyWeimer_d1'] = vWeimer_yhat*yhat_d1[hereinds]
            df.loc[hereinds,'ViyWeimer_d2'] = vWeimer_yhat*yhat_d2[hereinds]
            
        with pd.HDFStore(masterhdfdir+masterhdf, 'a') as store:
            for column in ['ViyWeimer_d1','ViyWeimer_d2']:
                store.append(column, df[column], format='t')
        
        # goodinds = np.isfinite(df['Viy_d2']) & np.isfinite(df['ViyWeimer_d2'])
        
        # goodinds = np.isfinite(df['Viy_d2']) & np.isfinite(df['ViyWeimer_d2']) & (df['Quality_flags'] == 4)


