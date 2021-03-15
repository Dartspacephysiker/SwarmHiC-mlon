#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 08:21:15 2020

@author: jone
"""


import numpy as np
from scipy.linalg import lstsq
import datetime as dt
import pandas as pd
from pysymmetry.visualization import grids
from pysymmetry.models import igrf
from secsy import utils as secs
import pickle
from scipy.stats import binned_statistic
import gc
try:
    from ..utils import rotate_ocb
except:
    from utils import rotate_ocb
#import rotate_ocb


def binning(data, filename, ocb='ampere_simple', LOWLAT=60, hemi='north', corot='', 
                EDI=False, workstation=False):

    """
    

    Parameters
    ----------
    data : Pandas DataFrame
        Holding the observational data
    filename : TYPE str
        used for saving the binned average dict. Also describes the selection parameters appled
    ocb : str
        which, if any (''), ocb normalization scheme to use. The default is 'ampere_simple'.
        Other options inclute ('ampere', 'ocbnorm') which both uses the ocbpy package. This functionality is 
        placed on hold until we discover why ocbpy has some wierd behaviour in its vector transformations. 
        ('ocb') uses the same simple normalization as 'ampere_simple', but uses IMAGE FUV boundaries from BAS database
    LOWLAT : int
        Lowest latitude for the binning. The default is 60.
    hemi : str, optional
        DESCRIPTION. The default is 'north'.
    corot : str, optional
        either '_corot' or '' specifying is one should add corot speed to velocities The default is ''.
    EDI : boolean, optional
        Whether the data is from the EDI instrument or not. The default is False.       

    Returns
    -------
    None.

    """
    
     
    #Prepare the data locations in OCB coords
    if ocb != '':
        if (ocb == 'ocb') | (ocb == 'ocbnorm'):
            boundary = 90. - np.median(data['oval_r'][np.isfinite(data['oval_r'])]) #71 #Where to locate the OCB
        if (ocb == 'ampere') | (ocb == 'ampere_simple'):
            use = data.quality>=0.15
            data = data[use].copy()
            ocboffsetmean = 3.34
            #boundary = (90. - np.median(data['oval_r'][np.isfinite(data['oval_r'])])) + ocboffsetmean #71 #Where to locate the OCB 3.34 is the mean of the A. Burrell function (2020 Anngeo paper)
            boundary = 75.
            #ocboffset = 3.34
            ocboffset = 4.01 * (1. - 0.55**2) / (1 + 0.55 * np.cos((np.radians(data.mlt*15.) + 0.92)))
            data.loc[:,'oval_r'] = data.oval_r - ocboffset
            if ocb == 'ampere_simple': #more simplistic normalization, not using ocbpy
                data.loc[:,'ocb'] = 90. - data.oval_r 
        if ocb == 'ocbnorm':
            use = (data['n_mlt_ocb']>=7) & (data['oval_r']<=23) & (data['oval_r']>=10) & (data['ovalcenter_r']<=8)  #Throw away data that will not be used
            data = data[use].copy()
        #Find the OCB normalized coordinates
        if (ocb == 'ocbnorm') | (ocb == 'ampere'):
            data.loc[:,'ocb_lat'] = np.nan
            data.loc[:,'ocb_mlt'] = np.nan
            if workstation:
                chunk = int(1e6) #1e6
            else:
                chunk = int(2e4) #1e6
            n = len(data)
            idxs = data.index.values
            if n > chunk:
                chunked = np.array_split(idxs, n/chunk)  
            else:
                chunked = [idxs]            
            ccc = 0
            print('Converting measurements into OCB frame (binning)')
            for chunk in chunked:
                data.loc[data.index.isin(chunk),'ocb_lat'], data.loc[data.index.isin(chunk),'ocb_mlt'] = \
                        rotate_ocb.normal_coord(data.loc[data.index.isin(chunk),'mlat'], \
                        data.loc[data.index.isin(chunk),'mlt'], data.loc[data.index.isin(chunk),'ovalcenter_phi'], \
                        data.loc[data.index.isin(chunk),'ovalcenter_r'], \
                        data.loc[data.index.isin(chunk),'oval_r'], \
                        boundary, hemi=1)
                ccc += 1
            print('Done')
            data = data.dropna() 
            #Discard observations too far from the ocb
            minlat = boundary - 10.     #Where to put E=0 constrain             
            use = (data.ocb_lat >= minlat)   
            data = data[use].copy()        
        if ocb == 'simple_edi': 
            a, b = pd.qcut(np.abs(data.mlat),10, retbins=True)
            #boundary = b[1]
            boundary = 74.5
            minlat = LOWLAT 
            #use = data.mlat >= boundary
            #data = data[use].copy()
            #print('Info: discard obsrvations at 10 % lowest latitudes (mlat < ' + str(boundary) + ')')
        if (boundary < 65) or (boundary > 82):
            print(3/0)
        if (ocb == 'ocb') | (ocb == 'ampere_simple'):
            minlat = boundary - 10.     #Where to put E=0 constrain
            use = (data.mlat >= minlat)   
            data = data[use].copy()              
    else:
        boundary = 74.
        minlat = LOWLAT   
        use = (data.mlat >= minlat)   
        data = data[use].copy()  

    #Make the binning grid
    M0 = 4  #4
    dr = 2  #2
    mlat_, mlt_, mltres_ = grids.equal_area_grid(dr = dr, M0 = M0, N = int((90-LOWLAT)//dr))
    ncells = len(mlt_)

    #Construct the binned average dict    
    HEIGHT = 300    #SD echo height in km
    grid = {'mlt':mlt_, 'mlat':mlat_, 'cmlt':mlt_ + mltres_/2., 'cmlat':mlat_ + dr/2.}
    grid['meanE'] = np.zeros(ncells)
    grid['meanV'] = np.zeros(ncells)
    grid['medianlos'] = np.zeros(ncells)
    grid['Ep'] = np.zeros(ncells)
    grid['Ee'] = np.zeros(ncells)
    grid['vp'] = np.zeros(ncells)
    grid['ve'] = np.zeros(ncells)    
    grid['n'] = np.zeros(ncells)
    grid['uniquehrs'] = np.zeros(ncells)
    grid['stdevE'] = np.zeros(ncells)
    grid['stdevV'] = np.zeros(ncells)
    grid['bias'] = np.zeros(ncells)
    grid['Bz'] = np.zeros(ncells)
    grid['By'] = np.zeros(ncells)
    grid['V'] = np.zeros(ncells)
    grid['AL'] = np.zeros(ncells)
    grid['tilt'] = np.zeros(ncells)
    data.loc[:,'mlat'] = np.abs(data.mlat)
    grid['boundary'] = np.zeros(ncells) + boundary
    grid['mltres'] = mltres_
    r0 = grid['mlat'][0]
    print('Computing binned averages on grid')
    for i in range(1,ncells+1): #start at 1
        #Determine mlat/mlt boundaries in present cell
        if i == ncells:
            mltstart = grid['mlt'][i-1]
            mltstop = 24.
            mlatstart = r0
            mlatstop = r0+dr            
        else:
            if grid['mlat'][i] == r0: 
                mltstart = grid['mlt'][i-1]
                mltstop = grid['mlt'][i]
            else:
                mltstart = grid['mlt'][i-1]
                mltstop = 24.        
            mlatstart = r0
            mlatstop = r0+dr
            r0 = mlat_[i]
        
        #Select data in the cell
        if ocb == 'ocbnorm':
            if (mlatstart < boundary) & (mlatstop > boundary):   #if boundary is within grid cell, only select observations on closed field lines
                intt = data[(data['ocb_mlt'] >= mltstart) & (data['ocb_mlt'] < mltstop) & \
                        (data['ocb_lat'] >= mlatstart) & (data['ocb_lat'] < boundary) & \
                        (data['n_mlt_ocb']>=7) & (data['oval_r']<=23) & (data['oval_r']>=10) & (data['ovalcenter_r']<=8) & \
                        (mlatstart > boundary-15)]
            else:   #all other grid cells
                intt = data[(data['ocb_mlt'] >= mltstart) & (data['ocb_mlt'] < mltstop) & \
                        (data['ocb_lat'] >= mlatstart) & (data['ocb_lat'] < mlatstop) & \
                        (data['n_mlt_ocb']>=7) & (data['oval_r']<=23) & (data['oval_r']>=10) & (data['ovalcenter_r']<=8) & \
                        (mlatstart > boundary-15)]
        if ocb == 'ampere':
            # if (mlatstart < boundary) & (mlatstop > boundary):   #if boundary is within grid cell, only select observations on closed field lines
            #     intt = data[(data['ocb_mlt'] >= mltstart) & (data['ocb_mlt'] < mltstop) & \
            #             (data['ocb_lat'] >= mlatstart) & (data['ocb_lat'] < boundary)]
            # else:   #all other grid cells
            intt = data[(data['ocb_mlt'] >= mltstart) & (data['ocb_mlt'] < mltstop) & \
                    (data['ocb_lat'] >= mlatstart) & (data['ocb_lat'] < mlatstop)]                    
        if (ocb == 'ocb') | (ocb == 'ampere_simple'):
            dmlatstart = boundary-mlatstop
            dmlatstop = boundary-mlatstart
            if (dmlatstart>0) or (dmlatstop>0):    #on closed field-lines
                intt = data[(data['mlt'] >= mltstart) & (data['mlt'] < mltstop) & \
                    (data['ocb']-data['mlat'] >= dmlatstart) & \
                    (data['ocb']-data['mlat'] < dmlatstop)]# & \
#                    (data['n_mlt_ocb']>=7) & (data['oval_r']<=23) & (data['oval_r']>=10) & (data['ovalcenter_r']<=8)]
            else:   #inside polar cap, do a normalization of polar cap size
                fracstart = np.abs(dmlatstop)/(90.-boundary)
                fracstop = np.abs(dmlatstart)/(90.-boundary)
                intt = data[(data['mlt'] >= mltstart) & (data['mlt'] < mltstop) & \
                    ((np.abs(data['mlat'])-np.abs(data['ocb']))/(90.-boundary) >= fracstart) & \
                    ((np.abs(data['mlat'])-np.abs(data['ocb']))/(90.-boundary) < fracstop)]# & \
#                    (data['n_mlt_ocb']>=7) & (data['oval_r']<=23) & (data['oval_r']>=10) & (data['ovalcenter_r']<=8)]            
        if (ocb == '') | (ocb == 'simple_edi'):   #(no ocb normalizing)
            intt = data[(data['mlt'] >= mltstart) & (data['mlt'] < mltstop) & \
                    (np.abs(data['mlat']) >= mlatstart) & (np.abs(data['mlat']) < mlatstop)]
            #print mltstart, mltstop, mlatstart, mlatstop, len(intt), i        
        if len(intt) <= 1:
            #print('not enough points in cell')
            continue
        
        #Now do the fitting of the data and calculate statistical metrics
        if EDI:
            grid['Ep'][i-1] = np.median(intt.EapexN)
            grid['Ee'][i-1] = np.median(intt.EapexE)
            grid['n'][i-1] = len(intt.index)
            grid['Bz'][i-1] = np.median(intt['Bz'])
            grid['V'][i-1] = np.median(intt['V'])
            grid['AL'][i-1] = np.median(intt['AL'])            
            if hemi == 'both':
                grid['tilt'][i-1] = np.median(np.abs(intt['tilt']))
                grid['By'][i-1] = np.median(np.abs(intt['By']))
            else:
                grid['tilt'][i-1] = np.median(intt['tilt'])
                grid['By'][i-1] = np.median(intt['By'])                
        else:
            az = np.array(intt['az'])
            if corot == '_corot':
                ve_echo = 2.*np.pi*(6473.+300.)*1000.*np.cos(np.deg2rad(intt['mlat']))/(3600.*24.)    #corotation is only in mag. east in MLT frame
                corot_along_los = np.sin(np.deg2rad(az))*ve_echo
                los = np.array(intt['vlos'] + corot_along_los)[np.newaxis].T
            else:
                los = np.array(intt['vlos'])[np.newaxis].T
            dV = los
            GV = np.array([np.sin(np.deg2rad(az)),np.cos(np.deg2rad(az))]).T
            v = lstsq(GV, dV, cond = 0.05)[0]
            #grid['stdevV'][i-1] = np.sqrt((1./(len(intt['vlos'])-1.)) * np.sum((los - GV.dot(v).flatten())**2))
            grid['meanV'][i-1] = np.sqrt(v[0]**2 + v[1]**2)
            grid['vp'][i-1] = v[1]
            grid['ve'][i-1] = v[0]            
            intt.loc[:,'B'] = np.zeros(len(intt['glat']))
            intt.loc[:,'year'] = intt.index.year
            years = intt['year'].unique()                
            for y in years:
                unique = intt['year'] == y
                glat = intt['glat'][unique]
                glon = intt['glon'][unique]
                time = dt.datetime(y,7,1)
                bnorth, beast, bdown = igrf(time, glat, glon, HEIGHT, geodetic=True)
                intt.loc[unique,'B'] = np.sqrt(bnorth.flatten()**2 + beast.flatten()**2 + bdown.flatten()**2) * 1e-9
            if (ocb == 'ocbnorm') | (ocb == 'ampere'):
                dE = (intt['oval_r'])/(90.-boundary) * intt['B']*los.flatten()  #Scale the obseved E during the known oval radius to the E corresponding to the average oval radius for the given conditions (boundary)
                losE_east = dE*np.cos(np.deg2rad(az)) #E-field component #2020-12-17 jreistad changed from sin, cos to cos, -sin
                losE_north = -dE*np.sin(np.deg2rad(az)) #E-field component
                ocb_e, ocb_n = rotate_ocb.rotate_ocb(intt['mlat'], intt['mlt'], intt['oval_r'], intt['ovalcenter_r'], intt['ovalcenter_phi']/15., losE_east, losE_north, boundary) #E-field components
                newaz_fast = np.rad2deg(np.arctan2(ocb_e,ocb_n)) #angle is referrigng to the los E-field, not vlos azimuth
                az = np.copy(newaz_fast)
            elif ocb == 'ampere_simple':
                dE = (intt['oval_r'])/(90.-boundary) * intt['B']*los.flatten()  #Scale the obseved E during the known oval radius to the E corresponding to the average oval radius for the given conditions (boundary)                
            else:
                dE = intt['B']*los.flatten()       #No scaling to polar cap size     
            #Now rotate observations into a new xy frame, where x is parallell to 12 MLT and y is toward dusk
            R = np.array([[np.cos(np.deg2rad(intt.mlt*15)),-np.sin(np.deg2rad(intt.mlt*15))], \
                           [np.sin(np.deg2rad(intt.mlt*15)),np.cos(np.deg2rad(intt.mlt*15))]])
            en = np.array([dE*np.cos(np.deg2rad(intt.az)), dE*-np.sin(np.deg2rad(intt.az))])     #xy components of dE in local magnetic frame
            x1y1 = np.einsum('ijk,jk->ik', R, en)   #Convert into frame aligned woth 12 MLT. Next is to swap axes
            xy = np.array([x1y1[1,:],-x1y1[0,:]])   #now x is parallell with 12 MLT, y is toward dusk
            xyaz = np.arctan2(xy[1,:],xy[0,:])      #azimut angle in the new fixed xy frame
            xyGE = np.array([np.cos(xyaz),np.sin(xyaz)]).T
            xyE = lstsq(xyGE, dE, cond = 0.05)[0]
            #Convert back to the local east-north coordinate system
            x1y1E = np.array([-xyE[1],xyE[0]])
            R = np.array([[np.cos(np.deg2rad(grid['cmlt'][i-1]*15.)), -np.sin(np.deg2rad(grid['cmlt'][i-1]*15.))], \
                           [np.sin(np.deg2rad(grid['cmlt'][i-1]*15.)), np.cos(np.deg2rad(grid['cmlt'][i-1]*15.))]])
            Een = R.T.dot(x1y1E)
#            GV = np.array([np.sin(np.deg2rad(az)),np.cos(np.deg2rad(az))]).T
            #GE = np.array([np.cos(np.deg2rad(az)),-np.sin(np.deg2rad(az))]).T
            #print intt['oval_r']
            #E = lstsq(GE, dE, cond = COND)[0]
#            E2 = lstsq(GE, dE2, cond = COND)[0]
#            print (E-E2)*1000.
#            v = lstsq(GV, dV, cond = COND)[0]
            #grid['stdevE'][i-1] = np.sqrt((1./(len(intt['vlos'])-1.)) * np.sum((dE - GE.dot(E).flatten())**2))
#            grid['stdevV'][i-1] = np.sqrt((1./(len(intt['vlos'])-1.)) * np.sum((dV - GV.dot(E).flatten())**2))
            grid['meanE'][i-1] = np.sqrt(Een[0]**2 + Een[1]**2)
#            grid['meanV'][i-1] = np.sqrt(v[0]**2 + v[1]**2)
            grid['medianlos'][i-1] = np.median(intt['vlos'])
            grid['n'][i-1] = len(intt['vlos'])
            hr_datetime = [dt.datetime(kk.year, kk.month, kk.day,kk.hour)+dt.timedelta(hours=kk.minute //30) for kk in intt.index]
            grid['uniquehrs'][i-1] = pd.unique(hr_datetime).shape[0]
            grid['Ep'][i-1] = Een[1]
            grid['Ee'][i-1] = Een[0]
            grid['By'][i-1] = np.nanmedian(intt['By'])
            grid['Bz'][i-1] = np.nanmedian(intt['Bz'])
            grid['V'][i-1] = np.nanmedian(intt['V'])
            grid['AL'][i-1] = np.nanmedian(intt['AL'])            
            grid['tilt'][i-1] = np.nanmedian(intt['tilt'])   
            #estimate bias in look direction
            aztemp = az % 180.
            aznew = (aztemp * 2.)
            Gnew = np.array([np.sin(np.deg2rad(aznew)),np.cos(np.deg2rad(aznew))]).T    
            avg = np.mean(Gnew, axis=0)
            grid['bias'][i-1] = np.sqrt(avg[0]**2 + avg[1]**2)
            
        #some plotting for each bin to investigate validity of fit
        # if i==435:  #249
        #     import matplotlib.pyplot as plt
        #     fig=plt.figure(figsize=(10,10))
        #     ax = fig.add_subplot(111)
        #     #negs = az<0
        #     #az[negs] = az[negs] + 180.
        #     #intt.loc[negs,'vlos'] = -1.*intt[negs]['vlos']
        #     # I THINK ONE SHOULT SHIFT AZ BY -90 DEG WHEN PLOTTING THE LOS_E DUE TO THE WAY GE IS CONSTRUCTED (DUE TO E = -V X B)
        #     azplot = np.rad2deg(xyaz) #+ 90. #xyaz
        #     mmm = (azplot >= 180)
        #     azplot[mmm] = azplot[mmm]- 360.
        #     #Make az\in [0,180] but distinguish on sign of vlos
        #     nnn = (azplot<0)
        #     azplot[nnn] = azplot[nnn] + 180.
        #     dE[nnn] = dE[nnn]*-1.
        #     ax.scatter(azplot,dE*1000.,color='blue', s=2, alpha = 0.3)
        #     azs = np.linspace(0,180,360)            
        #     newG = np.array([np.cos(np.deg2rad(azs)),np.sin(np.deg2rad(azs))]).T
        #     pred_az = np.arctan2(xyE[1],xyE[0])*180./np.pi
        #     pred_los = newG.dot(xyE)
        #     pred_los0 = np.linalg.norm(xyE)#sqrt(xyE[0]**2+xyE[1]**2)
        #     if pred_az<0:
        #         pred_az = pred_az+180.
        #         pred_los0 = -1.*pred_los0
        #     ax.plot(azs,pred_los*1000., color='red',linewidth=4)
        #     ax.scatter(pred_az,pred_los0*1000.,color='red', s=500)
        #     #Plot binned average
        #     a,b,c=binned_statistic(azplot,dE,statistic='mean',bins=18)
        #     center = (b[0:-1] + b[1:])/2.
        #     d,e,f=binned_statistic(azplot,dE,statistic='std',bins=18)
        #     ax.errorbar(center,a*1000.,yerr=d*1000., color='orange')
        #     #ax.set_ylim([0,50])
        # #            ax.set_xlim([0,180])
        #     ax.set_xlim([0,180])
        #     ax.set_ylim([-40,40])
        #     ax.text(120,35,'mlat: [%2i, %2i]' % (mlatstart,mlatstop), fontsize=18)
        #     ax.text(120,31,'mlt: [%2.1f, %2.1f]' % (mltstart,mltstop), fontsize=18)
        #     #ax.text(10,1350,'mlt: ['+tr(float(mltmin))[0:4]+','+str(mltmax)[0:4]+']', fontsize=18)
        #     ppp = float(grid['stdevE'][i-1])*1000.
        #     #ax.text(120,65,'$\sigma$ = %3i mV/m' % ppp, fontsize=18)
        #     #ax.text(120,60,'bias = '+str(grid['bias'][i-1])[0:4], fontsize=18)
        #     Emag = np.sqrt(Een[0]**2+Een[1]**2)
        #     ppp = float(Emag)*1000.
        #     ax.text(120,27,'$|$E$|$ = %i mV/m' % ppp, fontsize=18)
        #     ax.text(120,23,'$\phi = %3i^{\circ}$' % pred_az, fontsize=18)
        #     ax.text(120,19,'$\#$ = '+str(len(intt['mlt'])), fontsize=18)
        #     ax.text(120,15,'Unique hrs: '+str(int(grid['uniquehrs'][i-1])), fontsize=18)
        #     #ax.text(10,1100,'iterations = '+str(counter), fontsize=18)
        #     ax.set_ylabel('Line-of-sight E [mV/m]')
        #     ax.set_xlabel('Angle with 12 MLT, $\phi [^{\circ}]$')
        #     for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        #         item.set_fontsize(20)
        #     fname = './plots/scatterplots/'+filename+'_i=%03i.png' % i
        #     fig.savefig(fname, bbox_inches='tight', dpi = 250)
        #     #print 3/0
        if (i % 100) == 0:
            print(i, ' of ', len(grid['cmlat']), gc.collect())
    
    
    with open('./grid_files/binned_dict_'+filename+'_'+ocb+corot+'.npy','wb') as picklefile: 
        pickle.dump(grid, picklefile, protocol=2)  
    
    return grid

def secs_on_grid(grid, COND=0.05, nlim = 10, biaslim = 0.9, NZEROS = 1.25, min_distance = 0.3, \
                 LOWLAT=60, ocb = '', hrslim = 100, latoffset=0, dr = 2, M0 = 4, l2scale = 1e-1):

    """
    

    Parameters
    ----------
    grid : TYPE dict
        DESCRIPTION. the binned average dict returned by binning function
    COND : TYPE, optional
        DESCRIPTION. SVD cutoff value in least squares fitting. The default is 0.05.
    boundary : TYPE float, optional
        DESCRIPTION. The default is 71.
    nlim : TYPE int, optional
        DESCRIPTION. Minimum number of observations within the cell to be considered. The default is 10.
    biaslim : TYPE float, optional
        DESCRIPTION. The default is 0.9.
    NZEROS : TYPE, optional
        DESCRIPTION. The fraction of total number of grid cells that should be used as a boundry constrain
        at LOWLAT to force the solution to zero. The default is 1.25.
    min_distance : TYPE, optional
        DESCRIPTION. TAngular distance [degrees] between node/data to be considered unaffected by singularity issue. The default is 0.3, but depend on grid density.
        This argument is now implemented in the construction of the G matrix such that we use the modified functions 
        (eq 2.43 in Vanhamaki and Juussola 2020) when observations are closer than this limit.. The default is 0.3. 
    LOWLAT : TYPE int, optional
        DESCRIPTION. Lowest latitude to consider in further analysis. The default is 60.
    ocb : TYPE, optional
        DESCRIPTION. Which (if none, ='') ocb scaling method to use. The default is ''.
    hrslim : TYPE, optional
        DESCRIPTION. Minimum number of unique hours to have observation from to be considered in the further analysis. The default is 10.
    latoffset : TYPE, optional
        DESCRIPTION. The default is 0.
    dr : TYPE, optional
        DESCRIPTION. lat resolution on SECS grid. The default is 2.
    M0 : TYPE, optional
        DESCRIPTION. MLT resolutuin on SECS grid. The default is 4.    
    l2: TYPE, optional
        DESCRIOTION. 2-norm regularization parameter

    Returns
    -------
    None.

    """
    
#Fit potential to SD averaged E-field map
#Use the potential to look at convection speed. This will take into account the Earths magnetic field strength difference      
    #Initial parameters
    if ocb != '':
        #minlat = boundary - 15.     #Where to put E=0 constrain
        minlat = LOWLAT+min_distance+0.1 
    else:
        minlat = LOWLAT
    

    #####################################################
    # Make grid to do the analysis on, grid_cf, which is the same as the binned average grid passed to this function
    grid_cf = {}  
    iii = np.where((grid['cmlat'] >= LOWLAT))    
    grid_cf['mlat'] = grid['cmlat'][iii[0]]
    grid_cf['mlt'] = grid['cmlt'][iii[0]]
    grid_cf['mltres']  = grid['mltres'][iii[0]] 
    grid_cf['dr'] = dr
    
    
    ### DEFINE SECS GRID
    N = int((90-LOWLAT)/dr) #15 #60  Want this grid to be similar to the avaraged grid (black pins), only shifted in mlt by mltres_/2.
    mlat_, mlt_, mltres_ = grids.equal_area_grid(dr = dr, M0 = M0, N = N) #0.8,8,60, 2.5, 3, 60
    mlat_ = mlat_ + dr/2.
    iii = np.where((mlat_ >= minlat+latoffset))# & (mlt_ >= 6) & (mlt_ <= 18))
    grid_secc = {}
    grid_secc['mlat'] = mlat_[iii[0]]
    grid_secc['mlt']  = mlt_[iii[0]]
    grid_secc['mltres']  = mltres_[iii[0]]

    #####################################################
    #The averages grid. Remove bins and add "observations" of E=0 at equatorward boundary
    ttt = (grid['n'] > nlim) & (grid['bias']<biaslim) & (grid['cmlat']>=minlat+latoffset)
    for key in grid.keys():
        grid[key] = grid[key][ttt]
    if NZEROS != 0:
        NN = np.round(NZEROS*len(grid['mlat'])).astype(int)
        for key in grid.keys():
            grid[key] = np.append(grid[key],np.zeros(NN))
        grid['cmlat'][-NN:] = minlat-dr/2
        grid['cmlt'][-NN:] = np.linspace(0,24,NN)
        grid['uniquehrs'][-NN:] = np.ones(NN)*hrslim
        grid['n'][-NN:] = np.ones(NN)*nlim
    ######
        grid['weights'] = np.ones(len(grid['n']))
        grid['weights'][-NN:] = 1.
        if hrslim != 0:
            crap = (grid['uniquehrs']<hrslim)
            grid['weights'][crap] = grid['uniquehrs'][crap]/hrslim
        else:
            crap = (grid['n']<nlim)
            grid['weights'][crap] = grid['n'][crap]/nlim            
    else:
        grid['weights'] = np.ones(len(grid['n']))
        if hrslim != 0:
            crap = (grid['uniquehrs']<hrslim)
            grid['weights'][crap] = grid['uniquehrs'][crap]/hrslim
        else:
            crap = (grid['n']<nlim)
            grid['weights'][crap] = grid['n'][crap]/nlim             
  
    #####################################################
    gridcmlat = grid['cmlat']
    gridcmlt = grid['cmlt']
    grid_secc['dr']  = dr    
    HEIGHT = 300            
    grid_secc['height']  = HEIGHT
    grid_secc['boundary']  = grid['boundary'][0]
    #Use the SECS technique to calculate the potential from the E-field derived from averaging SD LOS data
    #Implement the boundary condition on E only in East/West direction. We do not constrain the N-S component of E (the east/west returnflow)
#    Ge, Gptemp = secs.get_SECC_G_matrix(gridcmlat, gridcmlt, grid_secc['mlat'], grid_secc['mlt'], \
#                     constant = 1, HEIGHT=HEIGHT, type = 'curl_free')    
#    Getemp, Gp = secs.get_SECC_G_matrix(grid['cmlat'], grid['cmlt'], grid_secc['mlat'], grid_secc['mlt'], \
#                     constant = 1, HEIGHT=HEIGHT, type = 'curl_free')            
#    grid_secc['m'] = lstsq(np.vstack((np.multiply(Ge.T,np.append(grid['weights'],np.ones(NZEROS)).T).T, \
#                     np.multiply(Gp.T,grid['weights'].T).T)), np.hstack((np.append(grid['Ee'],np.zeros(NZEROS)) * \
#                     np.append(grid['weights'],np.ones(NZEROS)), grid['Ep']*grid['weights'])), cond = COND)[0] #value of the SECS nodes [charge]
#   If constraining boundary E field in both directions
    Ge, Gp = secs.get_SECS_J_G_matrices(gridcmlat, gridcmlt*15., grid_secc['mlat'], grid_secc['mlt']*15., \
                        constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + HEIGHT * 1e3, current_type = 'curl_free', \
                        singularity_limit=(6371.2 * 1e3 + HEIGHT * 1e3)*min_distance*np.pi/180.)        
    G = np.vstack((np.multiply(Ge.T,grid['weights'].T).T, \
                     np.multiply(Gp.T,grid['weights'].T).T))
    d = np.hstack((grid['Ee'] * grid['weights'], grid['Ep']*grid['weights']))
    GT = G.T
    GTG = GT.dot(G)
    GTd = GT.dot(d)
    scale = np.max(GTG)
    R = np.eye(GTG.shape[0]) * scale*l2scale
    #R = np.diag(np.ones(GTG.shape[0])*l2) #penalize large values of model parameter
    grid_secc['m'] = lstsq(GTG+R, GTd)[0] # minimize 2-norm of ((GTG.dot(m)+RI) - GTd)    
    # grid_secc['m'] = lstsq(np.vstack((np.multiply(Ge.T,grid['weights'].T).T, \
    #                  np.multiply(Gp.T,grid['weights'].T).T)), np.hstack((grid['Ee'] * \
    #                  grid['weights'], grid['Ep']*grid['weights'])), cond = COND)[0] #value of the SECS nodes [charge]    

    


    #####################################################        
    #Calculate the Curl Free E on the new grid_cf
    Ge, Gp = secs.get_SECS_J_G_matrices(grid_cf['mlat'], grid_cf['mlt']*15., grid_secc['mlat'], grid_secc['mlt']*15., \
                        constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + HEIGHT * 1e3, current_type = 'curl_free', \
                        singularity_limit=(6371.2 * 1e3 + HEIGHT * 1e3)*min_distance*np.pi/180.)      
    grid_cf['estimatedEe'] = Ge.dot(grid_secc['m'])
    grid_cf['estimatedEp'] = Gp.dot(grid_secc['m'])
    grid_cf['medianG'] = np.median(np.diag(Gp))
    
    #####################################################
    #Calculate potential of the curl free E-field #, now on the data-grid
    grid_cf['G_pot'] = secs.get_SECS_J_G_matrices(grid_cf['mlat'], grid_cf['mlt']*15, grid_secc['mlat'], grid_secc['mlt']*15, \
            constant = 1./(4.*np.pi), RI=6371.2 * 1e3 + HEIGHT * 1e3, current_type = 'potential')        
    grid_cf['psi'] = grid_cf['G_pot'].dot(grid_secc['m'])
    #Shift the potential to be close to 0 near LOWLAT
    bbb = (np.abs(grid_cf['mlat']-LOWLAT)<= 2)
    bbb_mean = np.mean(grid_cf['psi'][bbb])
    if np.isfinite(bbb_mean):
        print('Potential shift: ' + str(bbb_mean))
        grid_cf['psi'] = grid_cf['psi'] - bbb_mean
    return [grid_cf,grid_secc]    