"""
    1) Calculate mlat/MLT for each point, and the modified apex / quasi-dipole vector components
    2) Calculate the geographic coordinates for the case that the field is a dipole - the longitude will be 15*mlt
    3) Project the magnetic components to geographic coordinates in dipole field
"""

import os
from apexpy import Apex
import pandas as pd
import numpy as np
from datetime import datetime
from pytt.earth import geodesy
from pyamps.mlt_utils import mlon_to_mlt

storefn = '/SPENCEdata/Research/database/SHEIC/data_v1_update.h5'

Re = 6371.009 # mean radius of the Earth from Emmert et al. 


final_columns = ['h', 'qdlon', 'alat110', 'qdlat', 'mlt', 'Be', 'Bn', 'Bu', 'f1e', 'f1n', 'f2e', 'f2n', 'd1e', 'd1n', 'd2e', 'd2n']


for satellite in ['SwarmA', 'SwarmB', 'SwarmC']:
    with pd.HDFStore(storefn, mode = 'r') as store:
         indata = store[satellite + '/raw_data']

    years = np.unique(indata.index.year)
    A110 = {year:Apex(year, refh = 110) for year in years}

    print ('\nprocessing data from %s' % satellite)

    for year in sorted(years):
        print('\n')
        print(year)
        """ 1 COMPUTE GEODETIC COORDINATES AND COMPONENTS
        ##############################################""" 
        print('computing geodetic coordinates and components')
        data = indata[str(year)].copy()
        gdlat, h, X, Z = geodesy.geoc2geod(90 - data.gclat.values, data.r_km.values, -data.N_gc.values, data.U_gc.values)
        
        data['h'] = h
        data['gdlat'] = gdlat
        data['gdlon'] = data.gclon.values
        data['Be']  = data.E_gc
        data['Bn']  = X
        data['Bu']  = -Z
        
        """ 2 COMPUTE M(110) COORDINATES
        ##############################################"""
        print('computing modified apex coordinates')
     
        data['alat110'], data['alon110'] = A110[year].geo2apex(data.gdlat, data.gdlon, data.h)
        data['mlt']                      = mlon_to_mlt(data['alon110'].values, data.index, year)
        
        """ 3 COMPUTE QD COORDINATES
        ##############################################"""
        print('computing QD coordinates')
            
        data['qdlat'], data['qdlon'] = A110[year].geo2qd(data.gdlat, data.gdlon, data.h)
    
    
        """ 4 COMPUTE BASE VECTOR COMPONENTS """
        print('computing base vector components')
        
         
        #B = np.vstack((data.Be.values, data.Bn.values, data.Bu.values))
    
        f1, f2, f3, g1, g2, g3, d1, d2, d3, e1, e2, e3 = A110[year].basevectors_apex(data.gdlat.values, 
                                                                                     data.gdlon.values, 
                                                                                     data.h.values, 
                                                                                     coords = 'geo')
    
        F = f1[0]*f2[1] - f1[1]*f2[0]

        data['f1e'] = f1[0]
        data['f1n'] = f1[1]
        data['f2e'] = f2[0]
        data['f2n'] = f2[1]
        data['d1e'] = d1[0]
        data['d1n'] = d1[1]
        data['d2e'] = d2[0]
        data['d2n'] = d2[1]

    
        with pd.HDFStore(storefn, mode = 'a') as store:
            store.append(satellite + '/apex_data', data[final_columns], data_columns = True)
    
    
