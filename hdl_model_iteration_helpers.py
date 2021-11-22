import numpy as np
from scipy.linalg import cholesky, cho_solve
import gc

from utils import nterms, SHkeys
from gtg_array_utils import weighted_GTd_GTG_array, expand_GTG_and_GTd

def itersolve(filename,NV,MV,NT,MT,NWEIGHTS,NEQ,regularization='sheic'):

    # make regularization matrix:

    # lambda_V = 0
    # lambda_T = 1.e5
    # lambda_T = 1.e4
    # lambda_T = 1.e-1
    lambda_T = 1.e-9
    lambda_T = 0.

    
    while True:
        # print( 'solving... with lambda_T = %s, lambda_V = %s' % (lambda_T, lambda_V))
        print( 'solving... with lambda_T = %s' % (lambda_T))
        try:
            # n_cos_V = SHkeys(NV, MV).setNmin(1).MleN().Mge(0).n
            # n_sin_V = SHkeys(NV, MV).setNmin(1).MleN().Mge(1).n
            n_cos_T = SHkeys(NT, MT).setNmin(1).MleN().Mge(0).n
            n_sin_T = SHkeys(NT, MT).setNmin(1).MleN().Mge(1).n
            GTd_GTG_num = np.load(filename)
            GTd, GTG = expand_GTG_and_GTd(GTd_GTG_num, NWEIGHTS, NEQ)
            
            if regularization == 'amps':
                nn = np.hstack((lambda_T * n_cos_T  * (n_cos_T  + 1.)/(2*n_cos_T + 1.), lambda_T * n_sin_T  * (n_sin_T  + 1.)/(2*n_sin_T + 1.), 
                                lambda_V * n_cos_V  * (n_cos_V  + 1.)                 , lambda_V * n_sin_V  * (n_sin_V  + 1.)                 )).flatten()
            elif regularization == 'sheic':

                # This is carry-over from AMPS
                # nn = np.hstack((lambda_T * n_cos_T  * (n_cos_T  + 1.)/(2*n_cos_T + 1.), lambda_T * n_sin_T  * (n_sin_T  + 1.)/(2*n_sin_T + 1.))).flatten()
            
                # Think this is the condition that we should use, since our potential matches that of Lowes (1966) and he just shows (n+1)((g^m_n)^2+(h^m_n)^2) terms
                nn = np.hstack((lambda_T * (n_cos_T  + 1.), lambda_T * (n_sin_T  + 1.))).flatten()

            nn = np.tile(nn, NWEIGHTS)
                     
            R = np.diag(nn)
            
            c = cholesky(GTG + R, overwrite_a = True, check_finite = False)
            model_vector = cho_solve((c, 0), GTd)
            break # success!
        except:
            lambda_T *= 10 # increase regularization parameter by one order of magnitude
            gc.collect()
            continue

    return model_vector


def iterhuber(array, meanlim = 0.5, k = 1.5):
    """ compute mean and std with huber weights iteratively - repeat until |updated mean - old mean| < meanlim
    """


    m, s = huber(array, k = k)
    while True:
        newm, s = huber(array, k = k, inmean = m, instd = s)
        if np.abs(newm - m) < meanlim:
            m = newm
            break
        else:
            m = newm

    return m, s


def huber(array, k = 1.5, inmean = None, instd = None):
    """ compute huber mean of the array, using the Huber coefficient k. 
        adopted from matlab code by Nils Olsen and Egil Herland
    """

    if inmean is None and instd is None:
        mean_bare = np.mean(array)
        std_bare  = np.std(array)
        norm_res  = (array - mean_bare)/std_bare
    else:
        norm_res = (array - inmean)/instd

    # Huber weights
    w = k/np.abs(norm_res)
    w[w > 1.] = 1.

    # Huber mean and std:
    hmean = np.sum(w*array)/np.sum(w)
    hstd = np.sum(w*np.abs(array - hmean))/np.sum(w)

    return hmean, hstd
