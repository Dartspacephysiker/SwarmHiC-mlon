import numpy as np
import dask.array as da
import gc
d2r = np.pi/180



REFRE = 6372.


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



def getG_poltorapex_dask(NT, MT, NV, MV, qlat, alat, phi, h, f1e, f1n, f2e, f2n, d1e, d1n, d2e, d2n, RR = REFRE, makenoise = False, toroidal_minlat = 0):
    """ all input arrays should be dask arrays with shape (N, 1), and with the same chunksize. """
    gc.collect()

    # generate spherical harmonic keys    
    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)
    keys['cos_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(0)
    keys['sin_V'] = SHkeys(NV, MV).setNmin(1).MleN().Mge(1)
    m_cos_V = da.from_array(keys['cos_V'].m, chunks = keys['cos_V'].m.shape)
    m_sin_V = da.from_array(keys['sin_V'].m, chunks = keys['sin_V'].m.shape)
    m_cos_T = da.from_array(keys['cos_T'].m, chunks = keys['cos_T'].m.shape)
    m_sin_T = da.from_array(keys['sin_T'].m, chunks = keys['sin_T'].m.shape)

    nV = da.hstack((da.from_array(keys['cos_V'].n, chunks = keys['cos_V'].n.shape), da.from_array(keys['sin_V'].n, chunks = keys['sin_V'].n.shape)))

    if makenoise: print( m_cos_V.shape, m_sin_V.shape, m_cos_T.shape, m_sin_T.shape)

    # generate Legendre matrices - first get dicts of arrays, and then stack them in the appropriate fashion
    if makenoise: print( 'Calculating Legendre functions. alat and qlat shapes and chunks:', alat.shape, qlat.shape, alat.chunks, qlat.chunks)
    legendre_T = alat.map_blocks(lambda x: get_legendre_arrays(NT, MT, 90 - x, keys['cos_T'], minlat = toroidal_minlat), dtype = alat.dtype, chunks = (alat.chunks[0], tuple([2*len(keys['cos_T'])])))
    legendre_V = qlat.map_blocks(lambda x: get_legendre_arrays(NV, MV, 90 - x, keys['cos_V']), dtype = qlat.dtype, chunks = (qlat.chunks[0], tuple([2*len(keys['cos_V'])])))
    P_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
    dP_cos_T = -legendre_T[:,  len(keys['cos_T']):]
    P_cos_V  =  legendre_V[:, :len(keys['cos_V']) ] # split
    dP_cos_V = -legendre_V[:,  len(keys['cos_V']):]
    if makenoise: print( 'P, dP cos_T and P, dP cos_V size and chunks', P_cos_T.shape, dP_cos_T.shape, P_cos_V.shape, dP_cos_V.shape)#, P_cos_T.chunks, dP_cos_T.chunks, P_cos_V.chunks, dP_cos_V.chunks
    P_sin_T  =  P_cos_T[ :, keys['cos_T'].m.flatten() != 0] 
    dP_sin_T =  dP_cos_T[:, keys['cos_T'].m.flatten() != 0]
    P_sin_V  =  P_cos_V[ :, keys['cos_V'].m.flatten() != 0]
    dP_sin_V =  dP_cos_V[:, keys['cos_V'].m.flatten() != 0]  
    if makenoise: print( 'P, dP sin_T and P, dP sin_V size and chunks', P_sin_T.shape, dP_sin_T.shape, P_sin_V.shape, dP_sin_V.shape, P_sin_T.chunks[0], dP_sin_T.chunks[1], P_sin_V.chunks[1], dP_sin_V.chunks[1])

    # trig matrices:
    cos_T  =  da.cos(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    sin_T  =  da.sin(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))
    cos_V  =  da.cos(phi * d2r * m_cos_V)#.rechunk((phi.chunks[0], m_cos_V.shape[1]))
    sin_V  =  da.sin(phi * d2r * m_sin_V)#.rechunk((phi.chunks[0], m_sin_V.shape[1]))
    dcos_T = -da.sin(phi * d2r * m_cos_T)#.rechunk((phi.chunks[0], m_cos_T.shape[1]))
    dsin_T =  da.cos(phi * d2r * m_sin_T)#.rechunk((phi.chunks[0], m_sin_T.shape[1]))
    dcos_V = -da.sin(phi * d2r * m_cos_V)#.rechunk((phi.chunks[0], m_cos_V.shape[1]))
    dsin_V =  da.cos(phi * d2r * m_sin_V)#.rechunk((phi.chunks[0], m_sin_V.shape[1]))

    if makenoise: print( cos_T.shape, sin_T.shape)

    cos_qlat   = da.cos(qlat * d2r)
    cos_alat   = da.cos(alat * d2r)

    sinI  = 2 * da.sin( alat * d2r )/da.sqrt(4 - 3*cos_alat**2)

    r  = RR + h
    Rtor  = RR/r

    F = f1e*f2n - f1n*f2e
    if makenoise: print( F.shape, F)


    # matrix with horizontal spherical harmonic functions in QD coordinates
    V        = da.hstack((P_cos_V * cos_V, P_sin_V * sin_V ))

    # matrices with partial derivatives in QD coordinates:
    dV_dqlon  = da.hstack(( P_cos_V * dcos_V * m_cos_V,  P_sin_V * dsin_V * m_sin_V ))
    dV_dqlat  = da.hstack((dP_cos_V *  cos_V          , dP_sin_V *  sin_V           ))

    # matrices with partial derivatives in MA coordinates:
    dT_dalon  = da.hstack(( P_cos_T * dcos_T * m_cos_T,  P_sin_T * dsin_T * m_sin_T))
    dT_dalat  = da.hstack((dP_cos_T *  cos_T          , dP_sin_T *  sin_T          ))

    # Toroidal field components
    B_T_e  =   -d1n * dT_dalon / cos_alat + d2n * dT_dalat / sinI
    B_T_n  =    d1e * dT_dalon / cos_alat - d2e * dT_dalat / sinI
    B_T_u  =    da.zeros(B_T_n.shape, chunks = B_T_n.chunks)

    # Poloidal field components:
    B_V_e = (-f2n / (cos_qlat * r) * dV_dqlon + f1n * dV_dqlat / r) * RR * Rtor ** (nV + 1)
    B_V_n = ( f2e / (cos_qlat * r) * dV_dqlon - f1e * dV_dqlat / r) * RR * Rtor ** (nV + 1)
    B_V_u = da.sqrt(F) * V  * (nV + 1) * Rtor ** (nV + 2)

    G     = da.hstack((da.vstack((B_T_e , 
                                 B_T_n , 
                                 B_T_u )),   da.vstack((B_V_e, 
                                                        B_V_n, 
                                                        B_V_u))
                     ))

    return G


