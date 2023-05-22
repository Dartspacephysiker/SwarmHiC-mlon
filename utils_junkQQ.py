import numpy as np
import dask.array as da
import gc
d2r = np.pi/180

REFRE = 6371.2                  # Earth radius, km

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


def get_R_arrays__symm(nmax, mmax, theta, keys,
                       zero_thetas = 90.-np.array([47.,-47.]),
                       schmidtnormalize = True,
                       negative_m = False,
                       minlat = 0,
                       return_full_P_and_dP=False):
    """ get_R_arrays, but make everything 
    """

    zeros_0 = zero_thetas
    zeros_1 = 90.+(90.-zero_thetas)
    R_T0 = get_R_arrays(nmax, mmax, theta, keys,
                        zero_thetas = zeros_0,
                        schmidtnormalize = schmidtnormalize,
                        negative_m = negative_m,
                        minlat = minlat,
                        return_full_P_and_dP=return_full_P_and_dP)

    R_T1 = get_R_arrays(nmax, mmax, theta, keys,
                        zero_thetas = zeros_1,
                        schmidtnormalize = schmidtnormalize,
                        negative_m = negative_m,
                        minlat = minlat,
                        return_full_P_and_dP=return_full_P_and_dP)


    return (R_T0+R_T1)//2


def getG_torapex_dask_analyticzeros(NT, MT, alat, phi, 
                                    Be3_in_Tesla,
                                    lperptoB_dot_e1, lperptoB_dot_e2,
                                    RR=REFRE,
                                    makenoise=False,
                                    toroidal_minlat=0,
                                    apex_ref_height=110,
                                    zero_lats=np.array([47.,-47.])):
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
    R_T = alat.map_blocks(lambda x: get_R_arrays(NT, MT, 90 - x, keys['cos_T'],
                                                 minlat = toroidal_minlat,
                                                 zero_thetas = 90.-zero_lats), dtype = alat.dtype, chunks = (alat.chunks[0], tuple([2*len(keys['cos_T'])])))

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


def getG_torapex_dask_analyticzeros__symm(NT, MT, alat, phi, 
                                          Be3_in_Tesla,
                                          lperptoB_dot_e1, lperptoB_dot_e2,
                                          RR=REFRE,
                                          makenoise=False,
                                          toroidal_minlat=0,
                                          apex_ref_height=110,
                                          zero_lats=np.array([47.,-47.])):
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
    R_T = alat.map_blocks(lambda x: get_R_arrays__symm(NT, MT, 90 - x, keys['cos_T'],
                                                       minlat = toroidal_minlat,
                                                       zero_thetas = 90.-zero_lats), dtype = alat.dtype, chunks = (alat.chunks[0], tuple([2*len(keys['cos_T'])])))

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


def make_model_coeff_txt_file_analyticzeros(coeff_fn,
                                            NT=65,MT=3,
                                            NV=0,MV=0,
                                            TRANSPOSEEM=False,
                                            PRINTOUTPUT=False):

    from datetime import datetime
    import sys
    import os
    from utils import nterms, SHkeys

    Nmin = 2

    # NT, MT = 65, 3
    # # NV, MV = 45, 3
    # NV, MV = 0, 0
    NEQ = nterms(NT, MT, NV, MV, Nmin=Nmin)
    
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
    
    assert NWEIGHTS in (1,3,19),f"Have not yet implemented make_model_coeff_txt_file.py for {NWEIGHTS} weights!"
    
    N_NUM = NEQ*(NEQ+1)//2*NWEIGHTS*(NWEIGHTS+1)//2 + NEQ*NWEIGHTS # number of unique elements in GTG and GTd (derived quantity - do not change)
    
    coeffs = np.load(os.path.join(coeffdir,coefffile))  # Shape should be NEQ*NWEIGHTS
    print("Coeffs array shape:", coeffs.shape[0])
    print("NEQ*NWEIGHTS      =", NEQ*NWEIGHTS)
    if coeffs.shape[0] == NEQ*NWEIGHTS:
        print("Good! These should be the same")
    else:
        assert 2<0,"You're going to run into trouble! coeffs in coeff_fn are wrong size"

    keys = {} # dictionary of spherical harmonic keys
    keys['cos_T'] = SHkeys(NT, MT).setNmin(Nmin).MleN().Mge(0)
    keys['sin_T'] = SHkeys(NT, MT).setNmin(Nmin).MleN().Mge(1)
    
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
# Spherical harmonic degree, order: 65, 3 (for T) (BUT starts at N=2!)
# 
# column names:"""
    dadzilla = dadzilla.replace("DTSTR",dtstring)
    dadzilla = dadzilla.replace("65, 3 (for T)",f"{NT}, {MT} (for T)")
    dadzilla = dadzilla.replace("(BUT starts at N=2!)",f"(BUT starts at N={Nmin}!)")

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
    

