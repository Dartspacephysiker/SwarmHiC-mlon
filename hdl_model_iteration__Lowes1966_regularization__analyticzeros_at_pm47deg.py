"""
0) NOTE: If wanting to use ALL measurements and derive FULL model, use doFINAL = True under "Select which type of model"
1) load data file and set up G0
2) load model vector and model values from iteration i
3) calculate huber weights
4) make system of equations (GTG, GTd)
5) solve - with the least possible regularization - and save model vector (i + 1)
6) calculate model values and save (i + 1)
7) go back to (2), unless ||model i + 1|| is close to ||model i||
"""

# 2021/11/22 WORKS!
# 2023/04/25 The big idea is to reduce the degree of the model by two so that it runs from NT = 2 to NT = 65 (instead of from NT = 0 to NT = 65)

import numpy as np
import dask.array as da
import gc
import time
import h5py
import sys
from scipy.linalg import cholesky, cho_solve
from dask.diagnostics import ProgressBar
# from utils import nterms, SHkeys, getG_torapex_dask, make_model_coeff_txt_file
from utils import nterms_analyticzeros, SHkeys, getG_torapex_dask_analyticzeros, make_model_coeff_txt_file
from gtg_array_utils import weighted_GTd_GTG_array, expand_GTG_and_GTd
from functools import reduce
# from hdl_model_iteration_helpers import itersolve, iterhuber

MACHINE = 'Kalles'
MACHINE = 'SpencersLaptop'

assert MACHINE in ['KallesTower','SpencersLaptop']

t0 = time.time()

if MACHINE == 'KallesTower':
    # Directory where SHEIC/SWIPE stuff is located on Kalle's machine:  /mnt/5fa6bccc-fa9d-4efc-9ddc-756f65699a0a/spencer/SHEIC
    masterhdfdir = '/scratch/spencer/SHEIC/'
elif MACHINE == 'SpencersLaptop':
    masterhdfdir = '/SPENCEdata/Research/database/SHEIC/'

DATAVERSION = 'v1'
DATAVERSION = 'v2'                                       # 2021/11/19
datafile       = masterhdfdir+f'modeldata_{DATAVERSION}_update.hdf5' # where the data are stored (see data_preparation/07_make_model_dataset.py)

## Select which type of model
dosmall = True
doonlynegbzsouth = False
doonlynegby = False
doonlyposby = False
doassortment = False
doalldptilt = False
doFINAL = False                 # Use ALL data, all model parameters


# do_modded_model = dosmall or doonlynegbzsouth or doonlynegby or doonlyposby or doassortment or doalldptilt
do_modded_model = doonlynegbzsouth or doonlynegby or doonlyposby or doassortment or doalldptilt

MODELSUFF = '_analyticzero_at_47deg'

# MODELVERSION = DATAVERSION+'onlyca'
MODELVERSION = DATAVERSION+'onlyca_mV_per_m_lillambda'+MODELSUFF

# 2021/11/20 TRY ALL MODEL PARAMS (still mV per m, just junking the unnecessary suffix)
MODELVERSION = DATAVERSION+'ALLPARAMS'+MODELSUFF

if doFINAL:
    MODELVERSION = DATAVERSION+'FINAL'+MODELSUFF
    # if MACHINE == 'KallesTower':
    if MACHINE == 'SpencersLaptop':
        assert 2 < 0,"You have set MACHINE == 'SpencersLaptop'. You should not be using a laptop to calculate the full model coefficients."


modded_subinds = None
if dosmall:
    MODELVERSION = MODELVERSION+'small'+MODELSUFF
    indlets = slice(0,1000000,100)
    ninds = np.arange(indlets.start,indlets.stop,indlets.step).size
    print(f"Doing smaller version of database consisting of {ninds} indices: [{indlets.start}:{indlets.stop}:{indlets.step}]")
elif doonlynegbzsouth:
    MODELVERSION = MODELVERSION+'BzNegNH'+MODELSUFF
    indfile = 'negbz_array_indices.txt'
elif doonlynegby:
    MODELVERSION = MODELVERSION+'ByNeg'+MODELSUFF
    indfile = 'negby_array_indices.txt'
elif doonlyposby:
    MODELVERSION = MODELVERSION+'ByPos'+MODELSUFF
    indfile = 'posby_array_indices.txt'
elif doassortment:
    # indlets = np.where((full['mlat'].abs() >= 45 ) & \
    #                    # (full['By'] >= 0) & \
    #                    (np.abs(full['tilt']) <= 10) & \
    #                    ((full['f107obs'] >= np.quantile(full['f107obs'],0.25)) & (full['f107obs'] <= np.quantile(full['f107obs'],0.75))))[0]
    MODELVERSION = MODELVERSION+'Sortiment'+MODELSUFF
    indfile = 'sortiment_array_indices.txt'
    modded_Nsubinds = 300000
elif doalldptilt:
    # 20211120 indices that include all dipole tilts, limited range of 
    indfile = 'alldptilt_array_indices.txt'
    modded_Nsubinds = 900000
    randomseednumber = 123
    MODELVERSION = MODELVERSION+f'Alldptilt_{randomseednumber:d}'+MODELSUFF

if do_modded_model:
    print(f"Loading indices from file '{indfile}' (see journal__20210825__find_out_what_data_was_used_for_model_coeffs_based_on_slice_0_1000000_100__ie_10k_total_points.py)")
    indlets = np.int64(np.loadtxt(masterhdfdir+indfile))
    print(f"Got {len(indlets)} indices from file '{indfile}'")

    if modded_Nsubinds is not None:
        # modded_Nsubinds = 300000
        print(f"Trimming loaded indices by grabbing {modded_Nsubinds} random indices")
        np.random.seed(randomseednumber)
        indlets = np.random.choice(indlets,modded_Nsubinds,replace=False)

print("******************************")
print(f"MODEL VERSION: {MODELVERSION}")
print("******************************")
print("")

prefix_GTd_GTG_fn    = masterhdfdir+'matrices/model_'+MODELVERSION+'GTG_GTd_array_iteration_'
prefix_model_fn      = masterhdfdir+'matrices/model_'+MODELVERSION+'_iteration_'
prefix_model_value   = masterhdfdir+'matrices/model_'+MODELVERSION+'_values_iteration_'
prefix_huber_weights = masterhdfdir+'matrices/model_'+MODELVERSION+'_huber_iteration_'


""" MODEL/CALCULATION PARAMETERS """
i = -1 # number for previous iteration

NT, MT = 65, 3
# NV, MV = 45, 3
NV, MV = 0, 0
NEQ = nterms_analyticzeros(NT, MT, NV, MV)

if doFINAL:
    NWEIGHTS = 19
    CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
    
else:

    if 'onlyca' in MODELVERSION:
        NWEIGHTS = 3
        CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, BEEFED UP 'CAUSE ONLY ONE WEIGHT
    elif 'noparms' in MODELVERSION:
        NWEIGHTS = 1
        CHUNKSIZE = 200 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, BEEFED UP 'CAUSE ONLY ONE WEIGHT
    else:
        NWEIGHTS = 19
        CHUNKSIZE = 2 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, REDUCED for my laptop
    
print(f"NWEIGHTS, CHUNKSIZE: {NWEIGHTS}, {CHUNKSIZE}")
K = 5 # how many chunks shall be calculated at once

N_NUM = NEQ*(NEQ+1)//2*NWEIGHTS*(NWEIGHTS+1)//2 + NEQ*NWEIGHTS # number of unique elements in GTG and GTd (derived quantity - do not change)


""" HELPER FUNCTIONS """
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

def itersolve(filename):

    # make regularization matrix:

    # lambda_V = 0
    # lambda_T = 1.e5
    lambda_T = 1.e4

    lambda_T = 1.e2
    
    while True:
        # print( 'solving... with lambda_T = %s, lambda_V = %s' % (lambda_T, lambda_V))
        print( 'solving... with lambda_T = %s' % (lambda_T))
        try:
            # n_cos_V = SHkeys(NV, MV).setNmin(1).MleN().Mge(0).n
            # n_sin_V = SHkeys(NV, MV).setNmin(1).MleN().Mge(1).n
            n_cos_T = SHkeys(NT, MT).setNmin(2).MleN().Mge(0).n
            n_sin_T = SHkeys(NT, MT).setNmin(2).MleN().Mge(1).n
            GTd_GTG_num = np.load(filename)
            GTd, GTG = expand_GTG_and_GTd(GTd_GTG_num, NWEIGHTS, NEQ)
            
            # nn = np.hstack((lambda_T * n_cos_T  * (n_cos_T  + 1.)/(2*n_cos_T + 1.), lambda_T * n_sin_T  * (n_sin_T  + 1.)/(2*n_sin_T + 1.), 
            #                 lambda_V * n_cos_V  * (n_cos_V  + 1.)                 , lambda_V * n_sin_V  * (n_sin_V  + 1.)                 )).flatten()
            # nn = np.hstack((lambda_T * n_cos_T  * (n_cos_T  + 1.)/(2*n_cos_T + 1.), lambda_T * n_sin_T  * (n_sin_T  + 1.)/(2*n_sin_T + 1.))).flatten()
            
            # Regularization based on total power in the electric field (Lowes, 1966)
            nn = np.hstack((lambda_T * (n_cos_T  + 1.), lambda_T  * (n_sin_T  + 1.))).flatten()

            nn = np.tile(nn, NWEIGHTS)
                     
            R = np.diag(nn)
            
            c = cholesky(GTG + R, overwrite_a = True, check_finite = False)
            model_vector = cho_solve((c, 0), GTd)
            break # success!
        except:
            lambda_T *= 5 # increase regularization parameter by a factor of five
            gc.collect()
            continue

    return model_vector




""" HELPER FUNCTIONS DONE """



##################################
# (1) Load data file and set up G0
##################################
f = h5py.File(datafile, 'r')['/data']

# make a 2D array, one row for each item, and make a dictionary to map between the row and its name
names = [item[0] for item in f.items()]
datamap = dict(zip(names, range(len(names))))

# breakpoint()

if do_modded_model:
    data = da.vstack((da.from_array(f[name][()][indlets], chunks = CHUNKSIZE) for name in names))
else:
    data = da.vstack((da.from_array(f[name], chunks = CHUNKSIZE) for name in names))
ND = data.size/len(datamap) # number of datapoints
print( '%s - loaded data - %s points across %s arrays (dt = %.1f sec)' % (time.ctime(), ND, len(datamap), time.time() - t0))

# G0 = getG_torapex_dask(NT, MT, NV, MV,
#                        data[datamap['qdlat'  ]].reshape((data.shape[1], 1)),
#                        data[datamap['alat110']].reshape((data.shape[1], 1)),
#                    15* data[datamap['mlt'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['h'      ]].reshape((data.shape[1], 1)),
#                        data[datamap['f1e'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['f1n'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['f2e'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['f2n'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['d1e'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['d1n'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['d2e'    ]].reshape((data.shape[1], 1)),
#                        data[datamap['d2n'    ]].reshape((data.shape[1], 1)))
# We don't need the poloidal stuff
import warnings
# warnings.warn("You have not modified getG_torapex_dask to make sure that you're calculating the right stuff!")
# warnings.warn("2021/09/01 You have modified getG_torapex_dask so that RR is left in km (and hopefully potential in kV)!")
warnings.warn("2021/09/01 You have modified getG_torapex_dask so that coeffs have units mV/m (I hope!)")
# warnings.warn("2021/09/02 Kalle says regularization is currently based on magnetic energy integrated over the entire globe/sphere. This can't be right for the electric potential, so we need to think about it. ('Vi må heller tenke på hva våre antagelser om potensialet er, og formulere dette matematisk (ikke lett!)')")

NT,MT = 65,3

# assert 2<0
# print("ENTERING THE NOTE ZONE")
# from utils import getG_torapex_dask_analyticzeros,SHkeys,get_legendre_arrays,get_P_Q_and_R_arrays, get_R_arrays
# import numpy as np
# import dask.array as da

# # zero_keys = {} # dictionary of spherical harmonic keys
# # zero_keys['cos_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(0)
# # zero_keys['sin_T'] = SHkeys(NT, MT).setNmin(1).MleN().Mge(1)

# keys = {} # dictionary of spherical harmonic keys
# keys['cos_T'] = SHkeys(NT, MT).setNmin(2).MleN().Mge(0)
# keys['sin_T'] = SHkeys(NT, MT).setNmin(2).MleN().Mge(1)

# test_thetas = np.array([60.,70.]) # deg
# test_thetas = test_thetas.reshape((len(test_thetas),1))
# toroidal_minlat = 0
# legendre_T = get_legendre_arrays(NT, MT, test_thetas, keys['cos_T'], minlat=toroidal_minlat)
# P_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
# dP_cos_T = -legendre_T[:,  len(keys['cos_T']):]

# zero_thetas = 90.-np.array([47., -47.]).reshape((2,1))
# three_thetas = 90.-np.array([47., -47., 0.])
# three_thetas = three_thetas.reshape((len(three_thetas),1))
# zero_T = get_legendre_arrays(NT, MT, zero_thetas, keys['cos_T'], return_full_P_and_dP=True)
# # magicP_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
# # magicdP_cos_T = -legendre_T[:,  len(keys['cos_T']):]
# zero_T_P = {key:zero_T[0][key] for key in keys['cos_T']}
# zero_T_dP = {key:zero_T[1][key] for key in keys['cos_T']}
# #Now need to figure out how to implement Q and R functions

# return_full_P_and_dP = False

# dickie = get_P_Q_and_R_arrays(NT, MT, three_thetas, keys['cos_T'],
#                               zero_thetas=zero_thetas,
#                               return_full_P_and_dP=return_full_P_and_dP)
# R_T = get_R_arrays(NT, MT, three_thetas, keys['cos_T'],
#                                   zero_thetas=zero_thetas,
#                                   return_full_P_and_dP=False)

# dickiefull = get_P_Q_and_R_arrays(NT, MT, three_thetas, keys['cos_T'],
#                                   zero_thetas=zero_thetas,
#                                   return_full_P_and_dP=True)
# P_cos_T  = dickie[0][:, :len(keys['cos_T']) ] # split
# dP_cos_T = dickie[0][:, :len(keys['cos_T']) ] # split

# Q_cos_T  = dickie[1][:, :len(keys['cos_T']) ] # split
# dQ_cos_T = dickie[1][:, :len(keys['cos_T']) ] # split

# R_cos_T  = dickie[2][:, :len(keys['cos_T']) ] # split
# dR_cos_T = dickie[2][:, :len(keys['cos_T']) ] # split

# magicP_cos_T  =  legendre_T[:, :len(keys['cos_T']) ] # split
# magicdP_cos_T = -legendre_T[:,  len(keys['cos_T']):]

# TEST THAT LAMBDAS WHERE WE'RE SUPPOSED TO HAVE ZEROS ARE ACTUALLY ZEROS
# dickiezero = get_P_Q_and_R_arrays(NT, MT, zero_thetas, keys['cos_T'],
#                               zero_thetas=zero_thetas,
#                                   # zero_keys=zero_keys['cos_T'],
#                                   return_full_P_and_dP=True)

# Pshouldnotbezero = np.array([dickiezero['P'][key][0] for key in dickiezero['P'].keys()]).ravel()
# Qshouldbezero = np.array([dickiezero['Q'][key][0] for key in dickiezero['Q'].keys()]).ravel()
# Rshouldbezero = np.array([dickiezero['R'][key][0] for key in dickiezero['R'].keys()]).ravel()
# assert np.all(np.isclose(Qshouldbezero,0))
# assert np.all(np.isclose(Rshouldbezero,0))
#     # for key in keys['cos_T']:
#     #     if key[0] >= 2:

# print("LEAVING THE NOTE ZONE")

G0 = getG_torapex_dask_analyticzeros(NT, MT, 
                       data[datamap['mlat'           ]].reshape((data.shape[1], 1)),
                   15* data[datamap['mlt'            ]].reshape((data.shape[1], 1)),
                       data[datamap['Be3_in_Tesla'   ]].reshape((data.shape[1], 1)),
                       # data[datamap['B0IGRF'         ]].reshape((data.shape[1], 1)),
                       # data[datamap['d10'            ]].reshape((data.shape[1], 1)),
                       # data[datamap['d11'            ]].reshape((data.shape[1], 1)),
                       # data[datamap['d22'            ]].reshape((data.shape[1], 1)),
                       # data[datamap['d20'            ]].reshape((data.shape[1], 1)),
                       # data[datamap['d21'            ]].reshape((data.shape[1], 1)),
                       # data[datamap['d22'            ]].reshape((data.shape[1], 1)),
                       data[datamap['lperptoB_dot_e1']].reshape((data.shape[1], 1)),
                       data[datamap['lperptoB_dot_e2']].reshape((data.shape[1], 1)))

G0 = G0.rechunk((G0.chunks[0], G0.shape[1]))
print( '%s - done computing G0 matrix graph. G0 shape is %s (dt = %.1f sec)' % (time.ctime(), G0.shape, time.time() - t0))

# data vector:
# d = da.hstack(tuple(data[datamap[key]] for key in ['Be', 'Bn', 'Bu']))
d = da.hstack(tuple(data[datamap[key]] for key in ['lperptoB_dot_ViyperptoB']))
d = d.reshape((d.size, 1)) # to column
print( '%s - made d vector (dt = %.1f sec)' % (time.ctime(), time.time() - t0))

# prepare static weights (0.5 for side-by side satellites)
s_weight = data[datamap['s_weight']]
# s_weight = da.hstack((s_weight, s_weight, s_weight)) # stack three times - one for each component
# s_weight = da.hstack((s_weight)) # stack once - one for each component

# prepare matrix of weights (the external parameters)

if NWEIGHTS > 1:
    # OLD (multiple weights)
    weights = da.vstack( tuple([data[datamap['w' + str(jj + 1).zfill(2)]] for jj in range(NWEIGHTS-1)]))
    weights = weights.astype(np.float32)
    weights = da.vstack((da.ones(weights.shape[1], chunks = weights[0].chunks), weights)) # add a 1 weight on top (c0)

elif NWEIGHTS == 1:
    # NEW (only constant weight term)
    weights = da.ones(data[datamap['w01']].shape[0],chunks = data[0].chunks)
    weights = weights.reshape((1,weights.shape[0]))

else:
    assert 2<0

# Reshape 'em
# weights = da.hstack((weights, weights, weights)).T # tile them and transpose, shape is (Nmeas*3, NWEIGHTS)
weights = weights.T  # tile them and transpose, shape is (Nmeas, NWEIGHTS)

# weights = da.hstack((weights)).T # tile them and transpose
weights = weights.rechunk((G0.chunks[0], NWEIGHTS))

# breakpoint()
print("Entering loop ...")
while True: # enter loop
    #########################################################################
    # (2) Load model vector, model values, and huber weights from iteration i
    #########################################################################
    
    if i != -1:
        model_i       = np.load(prefix_model_fn + str(i) + '.npy')     # model vector
        dm_i          = np.load(prefix_model_value + str(i) + '.npy')  # model values
        if i != 0:
            huber_weights = np.load(prefix_huber_weights + str(i) + '.npy').flatten()
        else:
            huber_weights = np.ones_like(dm_i) # huber weights in first iteration are just ones

        i += 1
    
        ############################
        # 3) calculate huber weights
        ############################
    
        residuals  = (dm_i - d.flatten().compute())   # weighted by satellite weights
    
        rms_misfit = np.sqrt(np.average(residuals**2, weights = s_weight.flatten() * huber_weights))
        print('%s - misfit in iteration %s was %.2f \n' % (time.ctime(), i, rms_misfit))
        huber_mean_residual, huber_std = iterhuber(residuals**2)
        sigma = np.sqrt(huber_mean_residual) # <- rms residual - huber mean
        huber_weights = 1.5 * sigma / np.abs(residuals)
        huber_weights[huber_weights > 1] = 1
        huber_weights = da.from_array(huber_weights[:, np.newaxis], chunks = d.chunks)
        np.save(prefix_huber_weights + str(i) + '.npy', huber_weights)
        print( 'mean huber weight: %.2f' % huber_weights.mean().compute())
        gc.collect()

    else:
        huber_weights = 1.
        model_i = np.array(0.)
        i += 1
    
    
    ########################################
    # 4) make system of equations (GTG, GTd)
    ########################################
    GTd_GTG = da.map_blocks(weighted_GTd_GTG_array, G0 * huber_weights * s_weight[:, np.newaxis],
                                                     d * huber_weights * s_weight[:, np.newaxis], 
                                                     weights, chunks = (1, N_NUM))
    ( '%s - made GTd_GTG matrix graph (dt = %.1f sec)\n' % (time.ctime(), time.time() - t0))
    
    t0 = time.time()
    print( '%s - ready to compute (dt = 0 min)' % time.ctime())
    
    GTd_GTG_num = np.zeros(N_NUM, dtype = np.float64) # initialize the array
    
    print( '%s - calculating in %s steps ' % (time.ctime(), GTd_GTG.numblocks[0]//K + 1))
    for jj in range(GTd_GTG.numblocks[0]//K + 1):
        t_start = time.time()
        gc.collect()
        if jj*K >= GTd_GTG.shape[0]:
            break
        with ProgressBar():
            GTd_GTG_num = GTd_GTG_num + GTd_GTG[jj*K:(jj+1)*K].sum(axis = 0).compute()
        print( '\r%s - %s/%s steps - approximately %.1f minutes left (dt = %.1f min)' % (time.ctime(), jj + 1, GTd_GTG.numblocks[0]/K, ((GTd_GTG.numblocks[0] - jj*K)/K) * (time.time() - t_start)/60, (time.time() - t0)/60))
        if jj % 10 == 0:
            np.save(prefix_GTd_GTG_fn + str(i) + '.npy', GTd_GTG_num)
            print( 'saved GTd_GTG_num in %s at i = %s' % (prefix_GTd_GTG_fn + str(i) + '.npy', jj))
    
    print ('\n%s - done computing - saving %s' % (time.ctime(), prefix_GTd_GTG_fn + str(i) + '.npy'))
    np.save(prefix_GTd_GTG_fn + str(i) + '.npy', GTd_GTG_num)
    
    ###################################################################################
    # 5) solve - with the least possible regularization - and save model vector (i + 1)
    ###################################################################################
    model_new = itersolve(prefix_GTd_GTG_fn + str(i) + '.npy')
    np.save(prefix_model_fn + str(i) + '.npy', model_new)
    print( 'saved new model in %s' % (prefix_model_fn + str(i) + '.npy'))
    
    ############################################
    # 6) calculate model values and save (i + 1)
    ############################################
    model_vectors = np.split(model_new, NWEIGHTS)
    model_vector  = da.from_array(model_new, chunks = (G0.shape[1]))
    
    dm = np.array([])
    for jj in range(len(G0.chunks[0])):
        gc.collect()
        gg = G0[jj*CHUNKSIZE:(jj+1)*CHUNKSIZE].compute()
        ww = weights[jj*CHUNKSIZE:(jj+1)*CHUNKSIZE].compute()
    
        dm_ = reduce(lambda x, y: x + y, ((gg * w[:, np.newaxis]).dot(m) for m, w in zip(model_vectors, ww.T)))
        dm = np.hstack((dm, dm_))
        print( '%s/%s, dt = %s' % (jj, len(G0.chunks[0]), time.time() - t0))
    
    np.save(prefix_model_value + str(i) + '.npy', dm)
    print( 'saved new model values in %s ' % (prefix_model_value + str(i) + '.npy'))
    
    ###################################################################
    # 7) go back to (2), unless ||model i + 1|| is close to ||model i||
    ###################################################################
    difference = np.linalg.norm(model_new.flatten() - model_i.flatten())
    if i > 0:
        if difference < 0.01 * np.linalg.norm(np.load(prefix_model_fn + '0.npy')):
            break
    print( 'starting next iteration')
    

coeff_fn = prefix_model_fn + str(i) + '.npy'
make_model_coeff_txt_file(coeff_fn,
                          NT=NT,MT=MT,
                          NV=NV,MV=MV,
                          TRANSPOSEEM=False,
                          PRINTOUTPUT=False)
print( 'done. DONE!!!')
