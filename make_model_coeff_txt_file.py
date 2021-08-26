#;; This buffer is for text that is not saved, and for Lisp evaluation.
#;; To create a file, visit it with C-x C-f and enter text in its buffer.

import numpy as np

TRANSPOSEEM = True
PRINTOUTPUT = False

coeffdir = '/SPENCEdata/Research/database/SHEIC/matrices/10k_points/'
coefffile = 'model_v1_values_iteration_3.npy'
coefffile = 'model_v1_iteration_3.npy'

if TRANSPOSEEM:
    outfile = coefffile.replace('.npy','_TRANSPOSE.txt')
    print("Making TRANSPOSE coefficient file")
else:
    outfile = coefffile.replace('.npy','.txt')

# Read .npy coeff file

import sys
sheicpath = '/home/spencerh/Research/SHEIC/'
if not sheicpath in sys.path:
    sys.path.append(sheicpath)

from utils import nterms, SHkeys, getG_torapex_dask

NT, MT = 65, 3
# NV, MV = 45, 3
NV, MV = 0, 0
NEQ = nterms(NT, MT, NV, MV)
NWEIGHTS = 19
CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
CHUNKSIZE = 2 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights
K = 5 # how many chunks shall be calculated at once


N_NUM = NEQ*(NEQ+1)//2*NWEIGHTS*(NWEIGHTS+1)//2 + NEQ*NWEIGHTS # number of unique elements in GTG and GTd (derived quantity - do not change)

coeffs = np.load(coeffdir+coefffile)  # Shape should be NEQ*NWEIGHTS
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
fmtstring = "{:2d} {:1d}"+" {:10.4g}"*38

dadzilla = """# Sherical harmonic coefficients for the Swarm HEmispherically resolved Ionospheric Convection (SHEIC) model
# Produced Aug 2021
#
# Based on Swarm convection measurements made between 2013-12 to 2020.
# Reference: Laundal et al., "Solar wind and seasonal influence on ionospheric currents", Journal of Geophysical Research - Space Physics, doi:10.1029/2018JA025387, 2018
#
# Coefficient unit: V(??)
# Apex reference height: 110 km
# Earth radius: 6371.2 km
#
# Spherical harmonic degree, order: 65, 3 (for T)
# 
# column names:"""

openstring = "{:s} {:s} "+"{:s} "*38
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


    tor_c_const,tor_c_sinca,tor_c_cosca,tor_c_epsilon,tor_c_epsilon_sinca,tor_c_epsilon_cosca,tor_c_tilt,tor_c_tilt_sinca,tor_c_tilt_cosca,tor_c_tilt_epsilon,tor_c_tilt_epsilon_sinca,tor_c_tilt_epsilon_cosca,tor_c_tau,tor_c_tau_sinca,tor_c_tau_cosca,tor_c_tilt_tau,tor_c_tilt_tau_sinca,tor_c_tilt_tau_cosca,tor_c_f107 = COSCOEFFS[coscount,:]

    if cosm > 0:

        tor_s_const,tor_s_sinca,tor_s_cosca,tor_s_epsilon,tor_s_epsilon_sinca,tor_s_epsilon_cosca,tor_s_tilt,tor_s_tilt_sinca,tor_s_tilt_cosca,tor_s_tilt_epsilon,tor_s_tilt_epsilon_sinca,tor_s_tilt_epsilon_cosca,tor_s_tau,tor_s_tau_sinca,tor_s_tau_cosca,tor_s_tilt_tau,tor_s_tilt_tau_sinca,tor_s_tilt_tau_cosca,tor_s_f107 = SINCOEFFS[sincount,:]

        sincount += 1

    else:
        tor_s_const,tor_s_sinca,tor_s_cosca,tor_s_epsilon,tor_s_epsilon_sinca,tor_s_epsilon_cosca,tor_s_tilt,tor_s_tilt_sinca,tor_s_tilt_cosca,tor_s_tilt_epsilon,tor_s_tilt_epsilon_sinca,tor_s_tilt_epsilon_cosca,tor_s_tau,tor_s_tau_sinca,tor_s_tau_cosca,tor_s_tilt_tau,tor_s_tilt_tau_sinca,tor_s_tilt_tau_cosca,tor_s_f107 = np.ones(19)*np.nan



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
    if PRINTOUTPUT:
        print(thisline)
    outf.write(thisline+'\n')

outf.close()

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
