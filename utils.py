import numpy as np
import dask.array as da
import gc
d2r = np.pi/180

REFRE = 6371.2                  # Earth radius, km

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


def nterms_analyticzeros(NT = 0, MT = 0, NVi = 0, MVi = 0, NVe = 0, MVe = 0):
    """ return number of coefficients in an expansion in real spherical harmonics of
        toroidal magnetic potential truncated at NT, MT
        poloidal magnetic potential truncated at NVi, MVi for internal sources
        poloidal magnetic potential truncated at NVe, MVe for external sources
    """

    return len(SHkeys(NT , MT ).setNmin(2).MleN().Mge(0)) + \
           len(SHkeys(NT , MT ).setNmin(2).MleN().Mge(1)) + \
           len(SHkeys(NVe, MVe).setNmin(2).MleN().Mge(0)) + \
           len(SHkeys(NVe, MVe).setNmin(2).MleN().Mge(1)) + \
           len(SHkeys(NVi, MVi).setNmin(2).MleN().Mge(0)) + \
           len(SHkeys(NVi, MVi).setNmin(2).MleN().Mge(1))


def get_legendre_arrays(nmax, mmax, theta, keys,
                        schmidtnormalize = True,
                        negative_m = False,
                        minlat = 0,
                        return_full_P_and_dP=False):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
        ***NOTE: The algorithm calculates P^m_n (μ) = P^m_n(cosθ) and dP^m_n/dθ, but we wish
                 to instead calculate dP^m_n/dλ = -dP^m_n/dθ. Hence the application of a 
                 negative sign to dP^m_n here.

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

    if return_full_P_and_dP:
        return P, dP

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 

    return np.hstack((Pmat, dPmat))


def get_P_Q_and_R_arrays(nmax, mmax, theta, keys,
                       zero_thetas = 90.-np.array([47.,-47.]),
                       # zero_keys = None,
                       schmidtnormalize = True,
                       negative_m = False,
                       minlat = 0,
                       return_full_P_and_dP=False):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
        ***NOTE: The algorithm calculates P^m_n (μ) = P^m_n(cosθ) and dP^m_n/dθ, but we wish
                 to instead calculate dP^m_n/dλ = -dP^m_n/dθ. Hence the application of a 
                 negative sign to dP^m_n here.

        must be tested for large n - this could be unstable
        sum over m should be 1 for all thetas

        Same as get_legendre, but returns a N by 2M array, where N is the size of theta,
        and M is the number of keys. The first half the columns correspond to P[n,m], with
        n and m determined from keys - an shkeys.SHkeys object - and the second half is dP[n,m]

        theta must be a column vector (N, 1)
    """

    assert len(zero_thetas) == 2
    # assert np.all([key in zero_keys for key in keys]),"'zero_keys' must include all keys in 'keys'!"

    # zero_keys = {} # dictionary of spherical harmonic keys
    # zero_keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    # zero_keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
    zero_keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0)


    zero_thetas = zero_thetas.reshape((2,1))
    zero_T = get_legendre_arrays(nmax, mmax, zero_thetas, zero_keys, return_full_P_and_dP=True)
    zero_T_P = {key:zero_T[0][key] for key in zero_keys}
    zero_T_dP = {key:zero_T[1][key] for key in zero_keys}

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

    iplus = 0
    iminus = 1
    Pratio_mu_plus = {key:zero_T_P[key][iplus][0]/zero_T_P[1,0][iplus][0] for key in zero_keys}
    Q = {key:P[key]-Pratio_mu_plus[key]*P[1,0] for key in zero_keys}
    dQ = {key:dP[key]-Pratio_mu_plus[key]*dP[1,0] for key in zero_keys}

    Q11_mu_minus = zero_T_P[1,1][iminus][0]-Pratio_mu_plus[1,1]*zero_T_P[1,0][iminus][0]

    R = {key:Q[key]*(1-Q[1,1]/Q11_mu_minus) for key in zero_keys}
    dR = {key:dQ[key]*(1-Q[1,1]/Q11_mu_minus)-Q[key]*dQ[1,1]/Q11_mu_minus for key in zero_keys}

    if return_full_P_and_dP:
        # return dict(P=P,dP=dP,
        #             Q=Q,dQ=dQ,
        #             R=R,dR=dR)
        return dict(P={key:P[key] for key in keys},
                    dP={key:dP[key] for key in keys},
                    Q=Q,dQ=dQ,
                    R=R,dR=dR)

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    Qmat  = np.hstack(tuple(Q[key] for key in keys))
    dQmat = np.hstack(tuple(dQ[key] for key in keys)) 
    Rmat  = np.hstack(tuple(R[key] for key in keys))
    dRmat = np.hstack(tuple(dR[key] for key in keys)) 

    return np.hstack((Pmat, dPmat)),np.hstack((Qmat, dQmat)),np.hstack((Rmat, dRmat))


def get_R_arrays(nmax, mmax, theta, keys,
                 zero_thetas = 90.-np.array([47.,-47.]),
                 # zero_keys = None,
                 schmidtnormalize = True,
                 negative_m = False,
                 minlat = 0,
                 return_full_P_and_dP=False):
    """ Schmidt normalization is optional - can be skipped if applied to coefficients 

        theta is colat [degrees]

        algorithm from "Spacecraft Attitude Determination and Control" by James Richard Wertz
        (http://books.google.no/books?id=GtzzpUN8VEoC&lpg=PP1&pg=PA781#v=onepage)
        ***NOTE: The algorithm calculates P^m_n (μ) = P^m_n(cosθ) and dP^m_n/dθ, NOT dP^m_n/dλ

        must be tested for large n - this could be unstable
        sum over m should be 1 for all thetas

        Same as get_legendre, but returns a N by 2M array, where N is the size of theta,
        and M is the number of keys. The first half the columns correspond to P[n,m], with
        n and m determined from keys - an shkeys.SHkeys object - and the second half is dP[n,m]

        theta must be a column vector (N, 1)
    """

    assert len(zero_thetas) == 2
    # assert np.all([key in zero_keys for key in keys]),"'zero_keys' must include all keys in 'keys'!"

    # zero_keys = {} # dictionary of spherical harmonic keys
    # zero_keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    # zero_keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
    zero_keys = SHkeys(nmax, mmax).setNmin(1).MleN().Mge(0)


    zero_thetas = zero_thetas.reshape((2,1))
    zero_T = get_legendre_arrays(nmax, mmax, zero_thetas, zero_keys, return_full_P_and_dP=True)
    zero_T_P = {key:zero_T[0][key] for key in zero_keys}
    zero_T_dP = {key:zero_T[1][key] for key in zero_keys}

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

    iplus = 0
    iminus = 1
    Pratio_mu_plus = {key:zero_T_P[key][iplus][0]/zero_T_P[1,0][iplus][0] for key in zero_keys}
    Q = {key:P[key]-Pratio_mu_plus[key]*P[1,0] for key in zero_keys}
    dQ = {key:dP[key]-Pratio_mu_plus[key]*dP[1,0] for key in zero_keys}

    Q11_mu_minus = zero_T_P[1,1][iminus][0]-Pratio_mu_plus[1,1]*zero_T_P[1,0][iminus][0]

    R = {key:Q[key]*(1-Q[1,1]/Q11_mu_minus) for key in zero_keys}
    dR = {key:dQ[key]*(1-Q[1,1]/Q11_mu_minus)-Q[key]*dQ[1,1]/Q11_mu_minus for key in zero_keys}

    if return_full_P_and_dP:
        # return dict(P=P,dP=dP,
        #             Q=Q,dQ=dQ,
        #             R=R,dR=dR)
        return dict(P={key:P[key] for key in keys},
                    dP={key:dP[key] for key in keys},
                    Q=Q,dQ=dQ,
                    R=R,dR=dR)

    Pmat  = np.hstack(tuple(P[key] for key in keys))
    dPmat = np.hstack(tuple(dP[key] for key in keys)) 
    Qmat  = np.hstack(tuple(Q[key] for key in keys))
    dQmat = np.hstack(tuple(dQ[key] for key in keys)) 
    Rmat  = np.hstack(tuple(R[key] for key in keys))
    dRmat = np.hstack(tuple(dR[key] for key in keys)) 

    return np.hstack((Rmat, dRmat))

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
                      RR=REFRE,
                      makenoise=False,
                      toroidal_minlat=0,
                      apex_ref_height=110):
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
    #NOTE: algorithm used by get_legendre_arrays calculates dP^m_n/dθ, but we wish
    #      to instead calculate dP^m_n/dλ = -dP^m_n/dθ. Hence the application of a 
    #      negative sign to dP^m_n here.
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

    # R = (RR + apex_ref_height)*1000                   # convert from km to m
    R = (RR + apex_ref_height)                   # DON'T convert from km to m; this way potential is in kV

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
    # lperptoB_dot_vperptoB = RR/(R * Be3_in_Tesla) * (lperptoB_dot_e2 / cos_alat * dT_dalon + \
    #                                                 lperptoB_dot_e1 / sinI     * dT_dalat)

    # Divide by a thousand so that model coeffs are in mV/m
    lperptoB_dot_vperptoB = RR/(R * Be3_in_Tesla * 1000) * (lperptoB_dot_e2 / cos_alat * dT_dalon + \
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


def getG_torapex_dask_analyticzeros(NT, MT, alat, phi, 
                                    Be3_in_Tesla,
                                    lperptoB_dot_e1, lperptoB_dot_e2,
                                    RR=REFRE,
                                    makenoise=False,
                                    toroidal_minlat=0,
                                    apex_ref_height=110,
                                    zerolat=47):
    """ all input arrays should be dask arrays with shape (N, 1), and with the same chunksize. """
    gc.collect()

    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(2).MleN().Mge(0)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(2).MleN().Mge(1)

    m_cos_T = da.from_array(keys['cos_T'].m, chunks = keys['cos_T'].m.shape)
    m_sin_T = da.from_array(keys['sin_T'].m, chunks = keys['sin_T'].m.shape)

    if makenoise: print( m_cos_T.shape, m_sin_T.shape)

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    if makenoise: print( 'Calculating Legendre functions. alat shape and chunks:', alat.shape, alat.chunks)
    R_T = alat.map_blocks(lambda x: get_R_arrays(NT, MT, 90 - x, keys['cos_T'], minlat = toroidal_minlat), dtype = alat.dtype, chunks = (alat.chunks[0], tuple([2*len(keys['cos_T'])])))

    R_cos_T  =  R_T[:, :len(keys['cos_T']) ] # split
    #NOTE: algorithm used by get_legendre_arrays within get_R_arrays calculates dP^m_n/dθ, but we wish
    #      to instead calculate dP^m_n/dλ = -dP^m_n/dθ and dR^m_n/dλ = -dR^m_n/dθ. Hence the application of a 
    #      negative sign to dR^m_n here.

    # Multiply by -1 because we want dR/dλ = -dR/dθ, and get_R_arrays calculates dR/dθ.
    dR_cos_T = -R_T[:,  len(keys['cos_T']):]

    if makenoise: print( 'R, dR cos_T size and chunks', R_cos_T.shape, dR_cos_T.shape)#, R_cos_T.chunks, dR_cos_T.chunks
    R_sin_T  =  R_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
    dR_sin_T =  dR_cos_T[:, keys['cos_T'].m.flatten() != 0]

    if makenoise: print( 'R, dR sin_T size and chunks', R_sin_T.shape, dR_sin_T.shape, R_sin_T.chunks[0], dR_sin_T.chunks[1])

    # trig matrices:
    cos_T  =  da.cos(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    sin_T  =  da.sin(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))

    dcos_T = -da.sin(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    dsin_T =  da.cos(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))

    if makenoise: print( cos_T.shape, sin_T.shape)

    cos_alat   = da.cos(alat * d2r)

    sinI  = 2 * da.sin( alat * d2r )/da.sqrt(4 - 3*cos_alat**2)

    R = (RR + apex_ref_height)                   # DON'T convert from km to m; this way potential is in kV

    # matrices with partial derivatives in MA coordinates:
    dT_dalon  = da.hstack(( R_cos_T * dcos_T * m_cos_T,  R_sin_T * dsin_T * m_sin_T))
    dT_dalat  = da.hstack((dR_cos_T *  cos_T          , dR_sin_T *  sin_T          ))

    # Divide by a thousand so that model coeffs are in mV/m
    lperptoB_dot_vperptoB = RR/(R * Be3_in_Tesla * 1000) * (lperptoB_dot_e2 / cos_alat * dT_dalon + \
                                                            lperptoB_dot_e1 / sinI     * dT_dalat)

    G = lperptoB_dot_vperptoB

    return G


def make_model_coeff_txt_file(coeff_fn,
                              NT=65,MT=3,
                              NV=0,MV=0,
                              TRANSPOSEEM=False,
                              PRINTOUTPUT=False):

    from datetime import datetime
    import sys
    import os
    from utils import nterms, SHkeys

    # NT, MT = 65, 3
    # # NV, MV = 45, 3
    # NV, MV = 0, 0
    NEQ = nterms(NT, MT, NV, MV)
    
    sheicpath = '/home/spencerh/Research/SHEIC/'
    if not sheicpath in sys.path:
        sys.path.append(sheicpath)
    
    dtstring = datetime.now().strftime("%d %B %Y")
    
    TRANSPOSEEM = False
    PRINTOUTPUT = False
    
    # coeffdir = '/SPENCEdata/Research/database/SHEIC/matrices/'
    # coefffile = '10k_points/model_v1_values_iteration_3.npy'
    # coefffile = '10k_points/model_v1_iteration_3.npy'
    
    coeffdir = os.path.dirname(coeff_fn)+'/'
    coefffile = os.path.basename(coeff_fn)
    
    # coeffdir = '/SPENCEdata/Research/database/SHEIC/matrices/'
    # coefffile = 'model_v1BzNegNH_iteration_4.npy'
    # coefffile = 'model_v1noparmsBzNegNH_iteration_2.npy'
    
    if TRANSPOSEEM:
        outfile = coefffile.replace('.npy','_TRANSPOSE.txt')
        print("Making TRANSPOSE coefficient file")
    else:
        outfile = coefffile.replace('.npy','.txt')
    
    # Read .npy coeff file
    print(f"Reading in {coefffile} for making a coeff .txt file ...")
    
    # CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
    if 'onlyca' in coefffile:
        print("This is a 'onlyca' coefffile with 3 weights ...")
        NWEIGHTS = 3
        CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, BEEFED UP 'CAUSE ONLY ONE WEIGHT
    elif 'noparms' in coefffile:
        print("This is a 'noparms' coefffile with 1 weight ...")
        NWEIGHTS = 1
        CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
    else:
        print("This is a coefffile with 19 weights ...")
        NWEIGHTS = 19
        CHUNKSIZE = 2 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights
    
    K = 5 # how many chunks shall be calculated at once
    
    assert NWEIGHTS in (1,3,19),f"Have not yet implemented make_model_coeff_txt_file.py for {NWEIGHTS} weights!"
    
    N_NUM = NEQ*(NEQ+1)//2*NWEIGHTS*(NWEIGHTS+1)//2 + NEQ*NWEIGHTS # number of unique elements in GTG and GTd (derived quantity - do not change)
    
    coeffs = np.load(os.path.join(coeffdir,coefffile))  # Shape should be NEQ*NWEIGHTS
    print("Coeffs array shape:", coeffs.shape[0])
    print("NEQ*NWEIGHTS      =", NEQ*NWEIGHTS)
    if coeffs.shape[0] == NEQ*NWEIGHTS:
        print("Good! These should be the same")
    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
    
    #len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(0)) #Out[30]: 257
    #len(SHkeys(NT , MT ).setNmin(1).MleN().Mge(1)) #Out[31]: 192
    #keys['cos_T'].n.shape #Out[33]: (1, 257)
    #keys['sin_T'].n.shape #Out[34]: (1, 192)
    #257+192 #Out[35]: 449 # == NEQ!
    
    COSN = keys['cos_T'].n.ravel()
    COSM = keys['cos_T'].m.ravel()
    SINN = keys['sin_T'].n.ravel()
    SINM = keys['sin_T'].m.ravel()
    
    ncosterms = len(COSN)
    nsinterms = len(SINN)
    
    # Based on Research/pySHEIC/pysheic/testem.py, it turns out that the order needs to be (NWEIGHTS, NEQ), followed by a transpose operation
    if TRANSPOSEEM:
        COEFFS = coeffs.reshape((NEQ,NWEIGHTS)).copy()
    else:
        COEFFS = coeffs.reshape((NWEIGHTS,NEQ)).T.copy()
        
    COSCOEFFS = COEFFS[:ncosterms,]
    SINCOEFFS = COEFFS[ncosterms:,]
    # fmtstring = "{:2d} {:1d}"+" {:10f}"*38
    fmtstring = "{:2d} {:1d}"+" {:10.4g}"*(NWEIGHTS*2)
    
    dadzilla = """# Spherical harmonic coefficients for the Swarm HEmispherically resolved Ionospheric Convection (SHEIC) model
# Produced DTSTR
#
# Based on Swarm convection measurements made between 2013-12 to 2020.
# Reference: Laundal et al., "Solar wind and seasonal influence on ionospheric currents", Journal of Geophysical Research - Space Physics, doi:10.1029/2018JA025387, 2018
#
# Coefficient unit: mV/m
# Apex reference height: 110 km
# Earth radius: 6371.2 km
#
# Spherical harmonic degree, order: 65, 3 (for T)
# 
# column names:"""
    dadzilla = dadzilla.replace("DTSTR",dtstring)
    dadzilla = dadzilla.replace("65, 3 (for T)",f"{NT}, {MT} (for T)")
    
    openstring = "{:s} {:s} "+"{:s} "*(NWEIGHTS*2)
    openstring = openstring.format('#n','m',
                           'tor_c_const'             ,  'tor_s_const'             ,
                           'tor_c_sinca'             ,  'tor_s_sinca'             ,
                           'tor_c_cosca'             ,  'tor_s_cosca'             ,
                           'tor_c_epsilon'           ,  'tor_s_epsilon'           ,
                           'tor_c_epsilon_sinca'     ,  'tor_s_epsilon_sinca'     ,
                           'tor_c_epsilon_cosca'     ,  'tor_s_epsilon_cosca'     ,
                           'tor_c_tilt'              ,  'tor_s_tilt'              ,
                           'tor_c_tilt_sinca'        ,  'tor_s_tilt_sinca'        ,
                           'tor_c_tilt_cosca'        ,  'tor_s_tilt_cosca'        ,
                           'tor_c_tilt_epsilon'      ,  'tor_s_tilt_epsilon'      ,
                           'tor_c_tilt_epsilon_sinca',  'tor_s_tilt_epsilon_sinca',
                           'tor_c_tilt_epsilon_cosca',  'tor_s_tilt_epsilon_cosca',
                           'tor_c_tau'               ,  'tor_s_tau'               ,
                           'tor_c_tau_sinca'         ,  'tor_s_tau_sinca'         ,
                           'tor_c_tau_cosca'         ,  'tor_s_tau_cosca'         ,
                           'tor_c_tilt_tau'          ,  'tor_s_tilt_tau'          ,
                           'tor_c_tilt_tau_sinca'    ,  'tor_s_tilt_tau_sinca'    ,
                           'tor_c_tilt_tau_cosca'    ,  'tor_s_tilt_tau_cosca'    ,
                           'tor_c_f107'              ,  'tor_s_f107'              )
    outf = open(coeffdir+outfile,'w')
    print("Opening "+coeffdir+outfile+' ...')
    if PRINTOUTPUT:
        print(dadzilla)
        print(openstring)
    outf.write(dadzilla+'\n')
    outf.write(openstring+'\n')
    
    coscount = 0
    sincount = 0
    for coscount in range(ncosterms):
        cosn = COSN[coscount]
        cosm = COSM[coscount]
    
    
        if NWEIGHTS == 19:
            
            # Get cos terms
            tor_c_const,tor_c_sinca,tor_c_cosca,tor_c_epsilon,tor_c_epsilon_sinca,tor_c_epsilon_cosca,tor_c_tilt,tor_c_tilt_sinca,tor_c_tilt_cosca,tor_c_tilt_epsilon,tor_c_tilt_epsilon_sinca,tor_c_tilt_epsilon_cosca,tor_c_tau,tor_c_tau_sinca,tor_c_tau_cosca,tor_c_tilt_tau,tor_c_tilt_tau_sinca,tor_c_tilt_tau_cosca,tor_c_f107 = COSCOEFFS[coscount,:]
            
            # Get sin terms
            if cosm > 0:
            
                tor_s_const,tor_s_sinca,tor_s_cosca,tor_s_epsilon,tor_s_epsilon_sinca,tor_s_epsilon_cosca,tor_s_tilt,tor_s_tilt_sinca,tor_s_tilt_cosca,tor_s_tilt_epsilon,tor_s_tilt_epsilon_sinca,tor_s_tilt_epsilon_cosca,tor_s_tau,tor_s_tau_sinca,tor_s_tau_cosca,tor_s_tilt_tau,tor_s_tilt_tau_sinca,tor_s_tilt_tau_cosca,tor_s_f107 = SINCOEFFS[sincount,:]
            
                sincount += 1
            
            else:
                tor_s_const,tor_s_sinca,tor_s_cosca,tor_s_epsilon,tor_s_epsilon_sinca,tor_s_epsilon_cosca,tor_s_tilt,tor_s_tilt_sinca,tor_s_tilt_cosca,tor_s_tilt_epsilon,tor_s_tilt_epsilon_sinca,tor_s_tilt_epsilon_cosca,tor_s_tau,tor_s_tau_sinca,tor_s_tau_cosca,tor_s_tilt_tau,tor_s_tilt_tau_sinca,tor_s_tilt_tau_cosca,tor_s_f107 = np.ones(NWEIGHTS)*np.nan
            
            # Make output line
            thisline = fmtstring.format(cosn,cosm,
                                        tor_c_const             ,  tor_s_const             ,
                                        tor_c_sinca             ,  tor_s_sinca             ,
                                        tor_c_cosca             ,  tor_s_cosca             ,
                                        tor_c_epsilon           ,  tor_s_epsilon           ,
                                        tor_c_epsilon_sinca     ,  tor_s_epsilon_sinca     ,
                                        tor_c_epsilon_cosca     ,  tor_s_epsilon_cosca     ,
                                        tor_c_tilt              ,  tor_s_tilt              ,
                                        tor_c_tilt_sinca        ,  tor_s_tilt_sinca        ,
                                        tor_c_tilt_cosca        ,  tor_s_tilt_cosca        ,
                                        tor_c_tilt_epsilon      ,  tor_s_tilt_epsilon      ,
                                        tor_c_tilt_epsilon_sinca,  tor_s_tilt_epsilon_sinca,
                                        tor_c_tilt_epsilon_cosca,  tor_s_tilt_epsilon_cosca,
                                        tor_c_tau               ,  tor_s_tau               ,
                                        tor_c_tau_sinca         ,  tor_s_tau_sinca         ,
                                        tor_c_tau_cosca         ,  tor_s_tau_cosca         ,
                                        tor_c_tilt_tau          ,  tor_s_tilt_tau          ,
                                        tor_c_tilt_tau_sinca    ,  tor_s_tilt_tau_sinca    ,
                                        tor_c_tilt_tau_cosca    ,  tor_s_tilt_tau_cosca    ,
                                        tor_c_f107              ,  tor_s_f107              )
    
        elif NWEIGHTS == 3:
            
            # Get cos terms
            tor_c_const,tor_c_sinca,tor_c_cosca = COSCOEFFS[coscount,:]
            
            # Get sin terms
            if cosm > 0:
            
                tor_s_const,tor_s_sinca,tor_s_cosca = SINCOEFFS[sincount,:]
            
                sincount += 1
            
            else:
                tor_s_const,tor_s_sinca,tor_s_cosca = np.ones(NWEIGHTS)*np.nan
            
            # Make output line
            thisline = fmtstring.format(cosn,cosm,
                                        tor_c_const             ,  tor_s_const             ,
                                        tor_c_sinca             ,  tor_s_sinca             ,
                                        tor_c_cosca             ,  tor_s_cosca             )
    
        elif NWEIGHTS == 1:
            
            # Get cos terms
            tor_c_const = COSCOEFFS[coscount,:][0]
            
            # Get sin terms
            if cosm > 0:
            
                tor_s_const = SINCOEFFS[sincount,:][0]
            
                sincount += 1
            
            else:
                tor_s_const = np.ones(NWEIGHTS)*np.nan
                tor_s_const = tor_s_const[0]
    
            # Make output line
            thisline = fmtstring.format(cosn,cosm,
                                        tor_c_const             ,  tor_s_const             )
    
        if PRINTOUTPUT:
            print(thisline)
        outf.write(thisline+'\n')
    
    outf.close()
    print("Made "+coeffdir+outfile)
    
    # tor_c_const               tor_s_const             
    # tor_c_sinca               tor_s_sinca             
    # tor_c_cosca               tor_s_cosca             
    # tor_c_epsilon             tor_s_epsilon           
    # tor_c_epsilon_sinca       tor_s_epsilon_sinca     
    # tor_c_epsilon_cosca       tor_s_epsilon_cosca     
    # tor_c_tilt                tor_s_tilt              
    # tor_c_tilt_sinca          tor_s_tilt_sinca        
    # tor_c_tilt_cosca          tor_s_tilt_cosca        
    # tor_c_tilt_epsilon        tor_s_tilt_epsilon      
    # tor_c_tilt_epsilon_sinca  tor_s_tilt_epsilon_sinca
    # tor_c_tilt_epsilon_cosca  tor_s_tilt_epsilon_cosca
    # tor_c_tau                 tor_s_tau               
    # tor_c_tau_sinca           tor_s_tau_sinca         
    # tor_c_tau_cosca           tor_s_tau_cosca         
    # tor_c_tilt_tau            tor_s_tilt_tau          
    # tor_c_tilt_tau_sinca      tor_s_tilt_tau_sinca    
    # tor_c_tilt_tau_cosca      tor_s_tilt_tau_cosca    
    # tor_c_f107                tor_s_f107              
    
