import numpy as np
import dask.array as da
import gc
d2r = np.pi/180

REFRE = 6372.

import pandas as pd
import numpy as np

def konveksjonsdatabase(sat,bare_substorm=False,
                        datadir='/SPENCEdata/Research/database/SHEIC/',
                        add_azimuth=False,
                        include_satellite_NEC_vectors=False,
                        stride=1,
                        min_quality=0,
                        for_fitting_Efield=True,
                        res_1s=False):
    #%%
    #Laste inn pakker, velge satellitt
    
    hdffile = f'Sat_{sat}_ct2hz_v0302_5sres.h5'
    
    if res_1s:
        print("1-s convection data ...")
        hdffile = hdffile.replace('5sres','1sres')

    if stride != 1:
        print(f"Decimating data by a factor of {stride}")

    #%%
    #Se HDFs innehold
    
    # print(f"Opening {hdffile}")
    # with pd.HDFStore(datadir+hdffile,'r') as store:
    #     print(store.keys())
    #     print("'/external' member contains:")
    #     print(store['/external'].columns)
        
    # Alle kolonner
    # ['/Bx', '/By', '/Bz',
    #  '/Ehx', '/Ehy', '/Ehz',
    #  '/Evx', '/Evy', '/Evz',
    #  '/Latitude', '/Longitude', '/Radius', 
    #  '/MLT', '/QDLatitude',
    #  '/Quality_flags',
    #  '/Vicrx', '/Vicry', '/Vicrz',
    #  '/Vixh', '/Vixh_error',
    #  '/Vixv', '/Vixv_error',
    #  '/Viy', '/Viy_d1', '/Viy_d2', '/Viy_error',
    #  '/Viz', '/Viz_error',
    #  '/VsatC', '/VsatE', '/VsatN',
    #  '/alt',
    #  '/d10', '/d11', '/d12', '/d20', '/d21', '/d22', '/d30', '/d31', '/d32',
    #  '/external',
    #  --> contains SW/substorm/tilt/F10.7: ['Bz', 'By', 'vx', 'nsw', 'f107obs', 'f107adj', 'tilt',
    #                                        'onset_dt_minutes', 'onset_mlat', 'onset_mlt']
    #  '/gdlat',
    #  '/mlat', '/mlon', '/mlt',
    #  '/yhat_d1', '/yhat_d2']    

    #%%
    #Velge ut noen kolonner og lage en DataFrame
    getcols = ['/mlt','/mlat','/alt','/yhat_d1','/yhat_d2','/Viy_d1','Viy_d2','/Quality_flags']
    getcols = ['/mlt','/mlat','/alt',
               '/yhat_d1','/yhat_d2',
               '/Viy_d1','Viy_d2',
               '/yhat_f1','/yhat_f2',
               '/Viy_f1','Viy_f2',
               '/Quality_flags']
    if include_satellite_NEC_vectors:
        getcols += ['/VsatN', '/VsatE', '/VsatC']

    if for_fitting_Efield:
        # getcols += ['/Viy','/Ehx','/Ehy','/gdlat','/Longitude','/Bx', '/By', '/Bz']
        getcols += ['/Viy','/gdlat','/Longitude','/Bx', '/By', '/Bz']

    getallextcols = True
    
    df = pd.DataFrame()
    with pd.HDFStore(datadir+hdffile,'r') as store:
        print("Getting these columns: "+", ".join(getcols)+' ...',end='')
        for col in getcols:
            if stride != 1:
                df[col.replace('/','')] = store[col][::stride]
            else:
                df[col.replace('/','')] = store[col]

        if for_fitting_Efield:
            print("Calculating B-field magnitude (in T, not nT!) and dropping measured Bx, By, and Bz ... ",end='')
            df['B'] = np.sqrt(df['Bx']**2+df['By']**2+df['Bz']**2)*1e-9
            df = df.drop(['Bx','By','Bz'],axis=1)

        print("Done!")
    
        if getallextcols:
            print("Getting solar wind, IMF, dipoletilt, F10.7, and substorm data ...",end='')
    
            if stride != 1:
                df = df.join(store['/external'][::stride])
            else:
                df = df.join(store['/external'])

            dfcols = list(df.columns)
            renamecols = dict(Bz='IMFBz',By='IMFBy')
            for key,rcol in renamecols.items():
                dfcols[np.where([key == col for col in dfcols])[0][0]] = rcol
            df.columns = dfcols
            print("Done!")
    #%%
    #Fjerne alle rader som ikke har Quality_flags >= 4 (disse er dårlig kalibrert)
    if min_quality > 0:
        print(f"Junking data with Quality_flags < {min_quality} ... ",end='')
        N = df.shape[0]
        good = df['Quality_flags'] >= min_quality
        df = df[good]
        print(f"Junked {N - df.shape[0]} rows and kept {df.shape[0]}")
    
    if bare_substorm:
        print("Only keeping rows associated with finite substorm onset parameters ...",end='')
        N = df.shape[0]
        good = np.isfinite(df['onset_mlt'])
        df = df[good]
        print(f"Junked {N - df.shape[0]} rows and kept {df.shape[0]}") 

        
    # Add az thing for bias calculation
    if add_azimuth:
        # From Richmond (1995) under Eq. (3.18):
        # "Figure 1 illustrates … d1 and e1 are more or less in the magnetic eastward direction;
        #  d2 and e2 are generally downward and/or equatorward;
        #  while d3 and e3 are along B_0."

        use_geo_coords = False

        df['az'] = np.nan

        if use_geo_coords:
            eastcomp = 'yhatE'
            northcomp = 'yhatN'

            print("Calculating azimuth angle using geographic coordinates, which I now (20210316) think is a bad idea ...")
            print(f"Adding azimuth using geographic coords (np.arctan2({eastcomp},{northcomp}))")

            xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=['VsatN','VsatE','VsatC'])
        
            df['yhatN'] = yhat[0,:]
            df['yhatE'] = yhat[1,:]
            
            df['az'] = np.rad2deg(np.arctan2(df['yhatE'].values,df['yhatN'].values))
        
        else:

            try:

                eastcomp = 'yhat_f1'
                northcomp = 'yhat_f2'
                
                print(f"Calculating azimuth angle using {eastcomp.replace('yhat_','')} and {northcomp.replace('yhat_','')} basis vectors in Apex coordinates," \
                      +" which I think is a better idea than using geographic coordinates ...")
                
                if northcomp == 'yhat_f2':
                    print("Making sure that yhat_f2 always points toward the pole in both hemispheres")
                    print("REMEMBER: We ALSO flip yhat_f2 in NH because of the way secs_direct and secs_binned handle az (i.e., they do nothing and are ignorant)")
                    print("The idea is to make sure that f2 always points poleward. According to Figure 5 in Richmond (1995), I guess this means we have to flip the sign of f2 in the SH")
                    df.loc[df['mlat'] > 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] > 0][eastcomp].values,df[df['mlat'] > 0][northcomp].values))
                    df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0][eastcomp].values,df[df['mlat'] < 0][northcomp].values*(-1)))
                    # df['az'] = np.rad2deg(np.arctan2(df[eastcomp].values,df[northcomp].values*(-1)))
                else:
                    assert 2<0,"What to do?"

            except:
                
                print("FAILED to use yhat_f1/f2 to calculate azimuth! Use d1/d2")

                eastcomp = 'yhat_d1'
                northcomp = 'yhat_d2'

                print(f"Calculating azimuth angle using {eastcomp.replace('yhat_','')} and {northcomp.replace('yhat_','')} basis vectors in Apex coordinates," \
                      +" which I think is a better idea than using geographic coordinates (but shouldn't we use e1 and e2 or f1 and f2?) ...")
        
                # print("Making sure that yhat_d2 always points toward the pole")
                # df['az'] = np.nan
                
                # df.loc[df['mlat'] > 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] > 0][eastcomp].values,df[df['mlat'] > 0][northcomp].values*(-1)))
                # # df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0][eastcomp].values,df[df['mlat'] < 0][northcomp].values))
                
                # print("REMEMBER: We ALSO flip yhat_d2 in SH because of the way secs_direct and secs_binned handle az (i.e., they do nothing)")
                # print("This way, ")
                # df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0][eastcomp].values,df[df['mlat'] < 0][northcomp].values*(-1)))
                    
                if northcomp == 'yhat_d2':
                    print("Making sure that yhat_d2 always points toward the pole in both hemispheres")
                    print("REMEMBER: We flip yhat_d2 in both hemis because of the way secs_direct and secs_binned handle az (i.e., they do nothing and are ignorant)")
                    # df.loc[df['mlat'] > 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] > 0][eastcomp].values,df[df['mlat'] > 0][northcomp].values*(-1)))
                    # df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0][eastcomp].values,df[df['mlat'] < 0][northcomp].values*(-1)))
                    df['az'] = np.rad2deg(np.arctan2(df[eastcomp].values,df[northcomp].values*(-1)))
                else:
                    assert 2<0,"What to do?"

    return df


def calc_az(df,vCol=['VsatN','VsatE','VsatC'],
            which_basevectors='apex_f',
            use_geo_coords=False):

    assert which_basevectors in ['apex_f','apex_d','geo']

    # Add az thing for bias calculation
    if which_basevectors == 'geo':
        print("Calculating azimuth angle using geographic coordinates, which I now (20210316) think is a bad idea ...")
        xhat, yhat, zhat = calculate_satellite_frame_vectors_in_NEC_coordinates(df,vCol=vCol)
    
        df['yhatN'] = yhat[0,:]
        df['yhatE'] = yhat[1,:]
        
        df['az'] = np.rad2deg(np.arctan2(df['yhatE'].values,df['yhatN'].values))
    
    elif which_basevectors in ['apex_f','apex_d']:
        df['az'] = np.nan
    
        if which_basevectors == 'apex_f':

            eastcomp = 'yhat_f1'
            northcomp = 'yhat_f2'
    
            print(f"Calculating azimuth angle using {eastcomp.replace('yhat_','')} and {northcomp.replace('yhat_','')} basis vectors in Apex coordinates," \
                  +" which I think is a better idea than using geographic coordinates ...")
        
            if northcomp == 'yhat_f2':
                print("Making sure that yhat_f2 always points toward the pole in both hemispheres")
                print("REMEMBER: We ALSO flip yhat_f2 in NH because of the way secs_direct and secs_binned handle az (i.e., they do nothing and are ignorant)")
                print("The idea is to make sure that f2 always points poleward. According to Figure 5 in Richmond (1995), I guess this means we have to flip the sign of f2 in the SH")
                df.loc[df['mlat'] > 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] > 0][eastcomp].values,df[df['mlat'] > 0][northcomp].values))
                df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0][eastcomp].values,df[df['mlat'] < 0][northcomp].values*(-1)))
                # df['az'] = np.rad2deg(np.arctan2(df[eastcomp].values,df[northcomp].values*(-1)))
            else:
                assert 2<0,"What to do?"
    
        elif which_basevectors == 'apex_d':

            eastcomp = 'yhat_d1'
            northcomp = 'yhat_d2'

            print("Calculating azimuth angle using d1 and d2 basis vectors in Apex coordinates, which I think is a better idea than using geographic coordinates (but shouldn't we use e1 and e2 or f1 and f2?) ...")
            
            print("Making sure that yhat_d2 always points toward the pole in both hemispheres")
    
            # df.loc[df['mlat'] > 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] > 0]['yhat_d1'].values,df[df['mlat'] > 0]['yhat_d2'].values*(-1)))
            # # df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0]['yhat_d1'].values,df[df['mlat'] < 0]['yhat_d2'].values))
            
            # print("REMEMBER: We ALSO flip yhat_d2 in SH because of the way secs_direct and secs_binned handle az (i.e., they do nothing)")
            # df.loc[df['mlat'] < 0,'az'] = np.rad2deg(np.arctan2(df[df['mlat'] < 0]['yhat_d1'].values,df[df['mlat'] < 0]['yhat_d2'].values*(-1)))
    
            df['az'] = np.rad2deg(np.arctan2(df[eastcomp].values,df[northcomp].values*(-1)))
    

def calculate_satellite_frame_vectors_in_NEC_coordinates(df,
                                                         vCol=['VsatN','VsatE','VsatC'],
                                                         assert_orthogonality=True):
    """
    Use the Swarm velocity components in NEC coordinates to calculate satellite-frame unit vectors in NEC coordinates

    ## Definition of satellite-track measurements according to "EFI TII Cross-Track Flow Data Release Notes" (Doc. no: SW-RN-UOC-GS-004, Rev: 6)
    "The coordinate system for the satellite-track measurements (along- and cross-track) is a right-handed orthonormal system
    * Defined in a frame of reference co-rotating with the Earth and having 
       * x in the direction of the satellite velocity, 
       * y perpendicular to x and horizontally to the right when facing the direction of motion, and 
       * z approximately downward completing the triad."

    "Measurements may be transformed into the north-east-centre (NEC) system using the supplied satellite velocity vector in NEC coordinates as a reference."

    2020-10-02
    SMH
    """

    def dotprod(a,b):
        return np.einsum('ij,ij->i',a,b)

    # Make sure we have all columns we need
    assert all(vCo in df.columns for vCo in vCol)

    vUnitCol = [vCo+'hat' for vCo in vCol]

    # Regne ut størrelsen til hastighetsvektorer
    
    VsatMag = np.sqrt((df[vCol]**2).sum(axis=1))
    # df['VsatMag'] = np.sqrt((df[vCol]**2).sum(axis=1))

    # Calculate magnitude of "horizontal" (i.e., not-vertical) component of satellite velocity vector
    # df['VsatHoriz'] = np.sqrt((df[['VsatN','VsatE']]**2).sum(axis=1))
    VsatHoriz = np.sqrt((df[['VsatN','VsatE']]**2).sum(axis=1))

    # The condition $|v_c|/v \ll \sqrt{v_n^2+v_e^2}/v$ must be fulfilled if the statement "z is approximately downward" is to be true
    # assert this condition
    # assert (np.abs(df['VsatC'])/df['VsatHoriz']).max() < 0.01
    assert (np.abs(df['VsatC'])/VsatHoriz).max() < 0.01

    # Regne ut komponentene til hastighets-enhetsvektor
    # for vCo,vUnitCo in zip(vCol,vUnitCol):
    #     # df[vUnitCo] = df[vCo]/df['VsatMag']
    #     df[vUnitCo] = df[vCo]/VsatMag

    # Forsikre oss om at enhetsvektorene har faktisk størrelse 1 :)
    # assert np.all(np.isclose(1,(df[vUnitCol]**2.).sum(axis=1)))

    #Da ifølge dokumentasjonen ...

    ########################################
    # i. Definere x-enhetsvektoren i NEC-koordinater
    
    # xN = df['VsatNhat']
    # xE = df['VsatEhat']
    # xC = df['VsatChat']
    
    xN = df['VsatN']/VsatMag
    xE = df['VsatE']/VsatMag
    xC = df['VsatC']/VsatMag

    assert np.all(np.isclose(1,xN**2+xE**2+xC**2))
    #                          (df[vUnitCol]**2.).sum(axis=1)
    # ))
    
    ########################################
    # ii. Definere y-enhetsvektoren i NEC-koordinater
    
    # Regne ut fortegn til yE.
    #Da y-enhetsvektoren er til høyre når man ser i bevegelsesretning, den peker østover (dvs yE > 0) når romfartøy går nordover, og vest (dvs yE < 0) når romfartøy går sørover
    yESign = np.int64((xN > 0) | np.isclose(xN,0))*2-1

    yE = yESign / np.sqrt( (xE**2 / xN**2) + 1)
    yN = - xE / xN * yE
    yC = yE * 0
    
    # Renorm just to make sure magnitudes are 1
    yMag = np.sqrt(yN**2 + yE**2 + yC**2)
    yN,yE,yC = yN / yMag,yE / yMag,yC / yMag
    
    ########################################
    # iii. Definere z-enhetsvektoren i NEC-koordinater

    zN, zE, zC = np.cross(np.vstack([xN,xE,xC]).T,np.vstack([yN,yE,yC]).T).T

    # Renorm just to make sure magnitudes are 1
    zMag = np.sqrt(zN**2+zE**2+zC**2)
    zN,zE,zC = zN / zMag,zE / zMag,zC / zMag
    
    xhat = np.vstack([xN,xE,xC])
    yhat = np.vstack([yN,yE,yC])
    zhat = np.vstack([zN,zE,zC])

    ########################################
    # Assert orthogonality
    
    if assert_orthogonality:
        # Assert xhat-prikk-yhat == 0
        assert np.max(np.abs(dotprod(xhat.T,yhat.T))) < 0.00001
        # Assert xhat-prikk-zhat == 0
        assert np.max(np.abs(dotprod(xhat.T,zhat.T))) < 0.00001
        # Assert yhat-prikk-zhat == 0
        assert np.max(np.abs(dotprod(yhat.T,zhat.T))) < 0.00001

    return xhat,yhat,zhat


class SHkeys(object):

    def __init__(self, Nmax, Mmax):
        """ container for n and m in spherical harmonics

            keys = SHkeys(Nmax, Mmax)

            keys will behave as a tuple of tuples, more or less
            keys['n'] will return a list of the n's
            keys['m'] will return a list of the m's
            keys[3] will return the fourth n,m tuple

            keys is also iterable

        """

        keys = []
        for n in range(Nmax + 1):
            for m in range(Mmax + 1):
                keys.append((n, m))

        self.keys = tuple(keys)
        self.make_arrays()

    def __getitem__(self, index):
        if index == 'n':
            return [key[0] for key in self.keys]
        if index == 'm':
            return [key[1] for key in self.keys]

        return self.keys[index]

    def __iter__(self):
        for key in self.keys:
            yield key

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def __str__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def setNmin(self, nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= nmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def Mge(self, limit):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= limit])
        self.make_arrays()
        return self

    def NminusModd(self):
        """ remove keys if n - m is even """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """ remove keys if n - m is odd """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """ add negative m to the keys """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))
        
        self.keys = tuple(keys)
        self.make_arrays()
        
        return self


    def make_arrays(self):
        """ prepare arrays with shape ( 1, len(keys) )
            these are used when making G matrices
        """

        if len(self) > 0:
            self.m = np.array(self)[:, 1][np.newaxis, :]
            self.n = np.array(self)[:, 0][np.newaxis, :]
        else:
            self.m = np.array([])[np.newaxis, :]
            self.n = np.array([])[np.newaxis, :]



def nterms(NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
    """ return number of coefficients in an expansion in real spherical harmonics of
        toroidal magnetic potential truncated at NT, MT
        poloidal magnetic potential truncated at NVi, MVi for internal sources
        poloidal magnetic potential truncated at NVe, MVe for external sources
    """

    return len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVe, MVe).setNmin(1).MleN().Mge(1)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(0)) + \
           len(SHkeys(NVi, MVi).setNmin(1).MleN().Mge(1))


def get_legendre_arrays(nmax, mmax, theta, keys, schmidtnormalize = True, negative_m = False, minlat = 0):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)

        must be tested for large n - this could be unstable
        sum over m should be 1 for all thetas

        Same as get_legendre, but returns a N by 2M array, where N is the size of theta,
        and M is the number of keys. The first half the columns correspond to P[n,m], with
        n and m determined from keys - an shkeys.SHkeys object - and the second half is dP[n,m]

        theta must be a column vector (N, 1)
    """


    P = {}
    dP = {}
    gc.collect()
    sinth = np.sin(d2r*theta)
    costh = np.cos(d2r*theta)

    if schmidtnormalize:
        S = {}
        S[0, 0] = 1.

    # initialize the functions:
    for n in range(nmax +1):
        for m in range(nmax + 1):
            P[n, m] = np.zeros_like(theta, dtype = np.float64)
            dP[n, m] = np.zeros_like(theta, dtype = np.float64)

    P[0, 0] = np.ones_like(theta, dtype = np.float64)
    P[0, 0][np.abs(90 - theta) < minlat] = 0
    for n in range(1, nmax +1):
        for m in range(0, min([n + 1, mmax + 1])):
            # do the legendre polynomials and derivatives
            if n == m:
                P[n, n]  = sinth * P[n - 1, m - 1]
                dP[n, n] = sinth * dP[n - 1, m - 1] + costh * P[n - 1, n - 1]
            else:

                if n == 1:
                    Knm = 0.
                    P[n, m]  = costh * P[n -1, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m]

                elif n > 1:
                    Knm = ((n - 1)**2 - m**2) / ((2*n - 1)*(2*n - 3))
                    P[n, m]  = costh * P[n -1, m] - Knm*P[n - 2, m]
                    dP[n, m] = costh * dP[n - 1, m] - sinth * P[n - 1, m] - Knm * dP[n - 2, m]

            if schmidtnormalize:
                # compute Schmidt normalization
                if m == 0:
                    S[n, 0] = S[n - 1, 0] * (2.*n - 1)/n
                else:
                    S[n, m] = S[n, m - 1] * np.sqrt((n - m + 1)*(int(m == 1) + 1.)/(n + m))


    if schmidtnormalize:
        # now apply Schmidt normalization
        for n in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, m]  *= S[n, m]
                dP[n, m] *= S[n, m]

    if negative_m:
        for n  in range(1, nmax + 1):
            for m in range(0, min([n + 1, mmax + 1])):
                P[n, -m]  = -1.**(-m) * factorial(n-m)/factorial(n+m) *  P[n, m]
                dP[n, -m] = -1.**(-m) * factorial(n-m)/factorial(n+m) * dP[n, m]

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 

    return np.hstack((Pmat, dPmat))



# def getG_poltorapex_dask(NT, MT, NV, MV, qlat, alat, phi, h, f1e, f1n, f2e, f2n, d1e, d1n, d2e, d2n, RR = REFRE, makenoise = False, toroidal_minlat = 0):
#     """ all input arrays should be dask arrays with shape (N, 1), and with the same chunksize. """
#     gc.collect()

#     # generate spherical harmonic keys    
#     keys = {} # dictionary of spherical harmonic keys
#     keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
#     keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
#     keys['cos_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(0)
#     keys['sin_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(1)
#     m_cos_V = da.from_array(keys['cos_V'].m, chunks = keys['cos_V'].m.shape)
#     m_sin_V = da.from_array(keys['sin_V'].m, chunks = keys['sin_V'].m.shape)
#     m_cos_T = da.from_array(keys['cos_T'].m, chunks = keys['cos_T'].m.shape)
#     m_sin_T = da.from_array(keys['sin_T'].m, chunks = keys['sin_T'].m.shape)

#     nV = da.hstack((da.from_array(keys['cos_V'].n, chunks = keys['cos_V'].n.shape), da.from_array(keys['sin_V'].n, chunks = keys['sin_V'].n.shape)))

#     if makenoise: print( m_cos_V.shape, m_sin_V.shape, m_cos_T.shape, m_sin_T.shape)

#     # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
#     if makenoise: print( 'Calculating Legendre functions. alat and qlat shapes and chunks:', alat.shape, qlat.shape, alat.chunks, qlat.chunks)
#     legendre_T = alat.map_blocks(lambda x: get_legendre_arrays(NT, MT, 90 - x, keys['cos_T'], minlat = toroidal_minlat), dtype = alat.dtype, chunks = (alat.chunks[0], tuple([2*len(keys['cos_T'])])))
#     legendre_V = qlat.map_blocks(lambda x: get_legendre_arrays(NV, MV, 90 - x, keys['cos_V']), dtype = qlat.dtype, chunks = (qlat.chunks[0], tuple([2*len(keys['cos_V'])])))
#     P_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
#     dP_cos_T = -legendre_T[:,  len(keys['cos_T']):]
#     P_cos_V  =  legendre_V[:, :len(keys['cos_V']) ] # split
#     dP_cos_V = -legendre_V[:,  len(keys['cos_V']):]
#     if makenoise: print( 'P, dP cos_T and P, dP cos_V size and chunks', P_cos_T.shape, dP_cos_T.shape, P_cos_V.shape, dP_cos_V.shape)#, P_cos_T.chunks, dP_cos_T.chunks, P_cos_V.chunks, dP_cos_V.chunks
#     P_sin_T  =  P_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
#     dP_sin_T =  dP_cos_T[:, keys['cos_T'].m.flatten() != 0]
#     P_sin_V  =  P_cos_V[ :, keys['cos_V'].m.flatten() != 0]
#     dP_sin_V =  dP_cos_V[:, keys['cos_V'].m.flatten() != 0]  
#     if makenoise: print( 'P, dP sin_T and P, dP sin_V size and chunks', P_sin_T.shape, dP_sin_T.shape, P_sin_V.shape, dP_sin_V.shape, P_sin_T.chunks[0], dP_sin_T.chunks[1], P_sin_V.chunks[1], dP_sin_V.chunks[1])

#     # trig matrices:
#     cos_T  =  da.cos(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
#     sin_T  =  da.sin(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))
#     cos_V  =  da.cos(phi * d2r * m_cos_V)#.rechunk((phi.chunks[0], m_cos_V.shape[1]))
#     sin_V  =  da.sin(phi * d2r * m_sin_V)#.rechunk((phi.chunks[0], m_sin_V.shape[1]))
#     dcos_T = -da.sin(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
#     dsin_T =  da.cos(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))
#     dcos_V = -da.sin(phi * d2r * m_cos_V)#.rechunk((phi.chunks[0], m_cos_V.shape[1]))
#     dsin_V =  da.cos(phi * d2r * m_sin_V)#.rechunk((phi.chunks[0], m_sin_V.shape[1]))

#     if makenoise: print( cos_T.shape, sin_T.shape)

#     cos_qlat   = da.cos(qlat * d2r)
#     cos_alat   = da.cos(alat * d2r)

#     sinI  = 2 * da.sin( alat * d2r )/da.sqrt(4 - 3*cos_alat**2)

#     r  = RR + h
#     Rtor  = RR/r

#     F = f1e*f2n - f1n*f2e
#     if makenoise: print( F.shape, F)


#     # matrix with horizontal spherical harmonic functions in QD coordinates
#     V        = da.hstack((P_cos_V * cos_V, P_sin_V * sin_V ))

#     # matrices with partial derivatives in QD coordinates:
#     dV_dqlon  = da.hstack(( P_cos_V * dcos_V * m_cos_V,  P_sin_V * dsin_V * m_sin_V ))
#     dV_dqlat  = da.hstack((dP_cos_V *  cos_V          , dP_sin_V *  sin_V           ))

#     # matrices with partial derivatives in MA coordinates:
#     dT_dalon  = da.hstack(( P_cos_T * dcos_T * m_cos_T,  P_sin_T * dsin_T * m_sin_T))
#     dT_dalat  = da.hstack((dP_cos_T *  cos_T          , dP_sin_T *  sin_T          ))

#     # Toroidal field components
#     B_T_e  =   -d1n * dT_dalon / cos_alat + d2n * dT_dalat / sinI
#     B_T_n  =    d1e * dT_dalon / cos_alat - d2e * dT_dalat / sinI
#     B_T_u  =    da.zeros(B_T_n.shape, chunks = B_T_n.chunks)

#     # Poloidal field components:
#     B_V_e = (-f2n / (cos_qlat * r) * dV_dqlon + f1n * dV_dqlat / r) * RR * Rtor ** (nV + 1)
#     B_V_n = ( f2e / (cos_qlat * r) * dV_dqlon - f1e * dV_dqlat / r) * RR * Rtor ** (nV + 1)
#     B_V_u = da.sqrt(F) * V  * (nV + 1) * Rtor ** (nV + 2)

#     G     = da.hstack((da.vstack((B_T_e , 
#                                  B_T_n , 
#                                  B_T_u )),   da.vstack((B_V_e, 
#                                                         B_V_n, 
#                                                         B_V_u))
#                      ))

#     return G


def getG_torapex_dask(NT, MT, alat, phi, 
                      Be3_in_Tesla,
                      # B0IGRF,
                      # d10,d11,d12,
                      # d20,d21,d22,
                      lperptoB_dot_e1, lperptoB_dot_e2,
                      RR = REFRE, makenoise = False, toroidal_minlat = 0, apex_ref_height=110):
    """ all input arrays should be dask arrays with shape (N, 1), and with the same chunksize. """
    gc.collect()

    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
    # keys['cos_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(0)
    # keys['sin_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(1)
    # m_cos_V = da.from_array(keys['cos_V'].m, chunks = keys['cos_V'].m.shape)
    # m_sin_V = da.from_array(keys['sin_V'].m, chunks = keys['sin_V'].m.shape)
    m_cos_T = da.from_array(keys['cos_T'].m, chunks = keys['cos_T'].m.shape)
    m_sin_T = da.from_array(keys['sin_T'].m, chunks = keys['sin_T'].m.shape)

    # D = np.sqrt( (d11*d22-d12*d21)**2 + \
    #              (d12*d20-d10*d22)**2 + \
    #              (d10*d21-d11*d20)**2)

    # Be3 = B0IGRF/D

    # nV = da.hstack((da.from_array(keys['cos_V'].n, chunks = keys['cos_V'].n.shape), da.from_array(keys['sin_V'].n, chunks = keys['sin_V'].n.shape)))

    # if makenoise: print( m_cos_V.shape, m_sin_V.shape, m_cos_T.shape, m_sin_T.shape)
    if makenoise: print( m_cos_T.shape, m_sin_T.shape)

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    if makenoise: print( 'Calculating Legendre functions. alat shape and chunks:', alat.shape, alat.chunks)
    legendre_T = alat.map_blocks(lambda x: get_legendre_arrays(NT, MT, 90 - x, keys['cos_T'], minlat = toroidal_minlat), dtype = alat.dtype, chunks = (alat.chunks[0], tuple([2*len(keys['cos_T'])])))
    # legendre_V = qlat.map_blocks(lambda x: get_legendre_arrays(NV, MV, 90 - x, keys['cos_V']), dtype = qlat.dtype, chunks = (qlat.chunks[0], tuple([2*len(keys['cos_V'])])))
    P_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
    dP_cos_T = -legendre_T[:,  len(keys['cos_T']):]
    # P_cos_V  =  legendre_V[:, :len(keys['cos_V']) ] # split
    # dP_cos_V = -legendre_V[:,  len(keys['cos_V']):]
    # if makenoise: print( 'P, dP cos_T and P, dP cos_V size and chunks', P_cos_T.shape, dP_cos_T.shape, P_cos_V.shape, dP_cos_V.shape)#, P_cos_T.chunks, dP_cos_T.chunks, P_cos_V.chunks, dP_cos_V.chunks
    if makenoise: print( 'P, dP cos_T size and chunks', P_cos_T.shape, dP_cos_T.shape)#, P_cos_T.chunks, dP_cos_T.chunks
    P_sin_T  =  P_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
    dP_sin_T =  dP_cos_T[:, keys['cos_T'].m.flatten() != 0]
    # P_sin_V  =  P_cos_V[ :, keys['cos_V'].m.flatten() != 0]
    # dP_sin_V =  dP_cos_V[:, keys['cos_V'].m.flatten() != 0]  
    # if makenoise: print( 'P, dP sin_T and P, dP sin_V size and chunks', P_sin_T.shape, dP_sin_T.shape, P_sin_V.shape, dP_sin_V.shape, P_sin_T.chunks[0], dP_sin_T.chunks[1], P_sin_V.chunks[1], dP_sin_V.chunks[1])
    if makenoise: print( 'P, dP sin_T size and chunks', P_sin_T.shape, dP_sin_T.shape, P_sin_T.chunks[0], dP_sin_T.chunks[1])

    # trig matrices:
    cos_T  =  da.cos(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    sin_T  =  da.sin(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))
    # cos_V  =  da.cos(phi * d2r * m_cos_V)#.rechunk((phi.chunks[0], m_cos_V.shape[1]))
    # sin_V  =  da.sin(phi * d2r * m_sin_V)#.rechunk((phi.chunks[0], m_sin_V.shape[1]))
    dcos_T = -da.sin(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    dsin_T =  da.cos(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))
    # dcos_V = -da.sin(phi * d2r * m_cos_V)#.rechunk((phi.chunks[0], m_cos_V.shape[1]))
    # dsin_V =  da.cos(phi * d2r * m_sin_V)#.rechunk((phi.chunks[0], m_sin_V.shape[1]))

    if makenoise: print( cos_T.shape, sin_T.shape)

    # cos_qlat   = da.cos(qlat * d2r)
    cos_alat   = da.cos(alat * d2r)

    sinI  = 2 * da.sin( alat * d2r )/da.sqrt(4 - 3*cos_alat**2)

    # r  = RR + h
    # Rtor  = RR/r

    R = (RR + apex_ref_height)*1000                   # convert from km to m

    # F = f1e*f2n - f1n*f2e
    # if makenoise: print( F.shape, F)


    # matrix with horizontal spherical harmonic functions in QD coordinates
    # V        = da.hstack((P_cos_V * cos_V, P_sin_V * sin_V ))

    # matrices with partial derivatives in QD coordinates:
    # dV_dqlon  = da.hstack(( P_cos_V * dcos_V * m_cos_V,  P_sin_V * dsin_V * m_sin_V ))
    # dV_dqlat  = da.hstack((dP_cos_V *  cos_V          , dP_sin_V *  sin_V           ))

    # matrices with partial derivatives in MA coordinates:
    dT_dalon  = da.hstack(( P_cos_T * dcos_T * m_cos_T,  P_sin_T * dsin_T * m_sin_T))
    dT_dalat  = da.hstack((dP_cos_T *  cos_T          , dP_sin_T *  sin_T          ))

    # things
    lperptoB_dot_vperptoB = 1/(R * Be3_in_Tesla) * (lperptoB_dot_e2 / cos_alat * dT_dalon + \
                                                    lperptoB_dot_e1 / sinI     * dT_dalat)

    # Toroidal field components
    # B_T_e  =   -d1n * dT_dalon / cos_alat + d2n * dT_dalat / sinI
    # B_T_n  =    d1e * dT_dalon / cos_alat - d2e * dT_dalat / sinI
    # B_T_u  =    da.zeros(B_T_n.shape, chunks = B_T_n.chunks)

    # Poloidal field components:
    # B_V_e = (-f2n / (cos_qlat * r) * dV_dqlon + f1n * dV_dqlat / r) * RR * Rtor ** (nV + 1)
    # B_V_n = ( f2e / (cos_qlat * r) * dV_dqlon - f1e * dV_dqlat / r) * RR * Rtor ** (nV + 1)
    # B_V_u = da.sqrt(F) * V  * (nV + 1) * Rtor ** (nV + 2)

    # G     = da.hstack((da.vstack((B_T_e , 
    #                              B_T_n , 
    #                              B_T_u ))
    # ))
    # G     = da.vstack((B_T_e , 
    #                    B_T_n , 
    #                    B_T_u ))
    G = lperptoB_dot_vperptoB

    return G

