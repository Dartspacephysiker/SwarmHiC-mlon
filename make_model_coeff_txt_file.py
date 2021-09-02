#;; This buffer is for text that is not saved, and for Lisp evaluation.
#;; To create a file, visit it with C-x C-f and enter text in its buffer.

import numpy as np
from datetime import datetime
dtstring = datetime.now().strftime("%d %B %Y")

TRANSPOSEEM = False
PRINTOUTPUT = False

# coeffdir = '/SPENCEdata/Research/database/SHEIC/matrices/'
# coefffile = '10k_points/model_v1_values_iteration_3.npy'
# coefffile = '10k_points/model_v1_iteration_3.npy'


coeffdir = '/SPENCEdata/Research/database/SHEIC/matrices/'
coefffile = 'model_v1BzNegNH_iteration_4.npy'

coefffile = 'model_v1noparmsBzNegNH_iteration_2.npy'

coefffile = 'model_v1noparms_mV_per_m_lillambdaBzNegNH_iteration_17.npy'

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

from utils import nterms, SHkeys

NT, MT = 65, 3
# NV, MV = 45, 3
NV, MV = 0, 0
NEQ = nterms(NT, MT, NV, MV)

# CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
if 'noparms' in coefffile:
    NWEIGHTS = 1
    CHUNKSIZE = 20 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights, KALLE'S ORIG
else:
    NWEIGHTS = 19
    CHUNKSIZE = 2 * NEQ * NWEIGHTS # number of spherical harmonics times number of weights

K = 5 # how many chunks shall be calculated at once

assert NWEIGHTS in (1,19),f"Have not yet implemented make_model_coeff_txt_file.py for {NWEIGHTS} weights!"

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
