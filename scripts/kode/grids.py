import numpy as np
d2r = np.pi / 180

def equal_area_grid(dr = 2, K = 0, M0 = 8, N = 20):
    """ function for calculating an equal area grid in polar coordinates

    Parameters
    ----------
    dr : float (optional)
        The latitudinal resolution of the grid. Default 2 degrees
    K : int (optional)
        This number determines the colatitude of the inner radius of the
        post poleward ring of grid cells (the pole is not inlcluded!). 
        It relates to this colatitude (r0) as r0/dr = (2K + 1)/2 => K = (2r0/dr - 1)/2.
        Default value is 0
    M0 : int (optional)
        The number of sectors in the most poleward ring. Default is 8
    N : int (optional)
        The number of rings to be included. This determiend how far 
        equatorward the grid extends. Typically if dr is changed from 2 to 1, 
        N should be doubled to reach the same latitude. Default is 20, which means
        that the equatorward edge of the grid is 89 - 20*2 = 49 degrees
        (the most poleward latitude is 89 with default values)

    Returns
    -------
    mlat : array
        Array of latitudes for the equatorward west ("lower left") corners of the grid
        cells.
    mlt : array
        Array of magnetic local times for the equatorward west ("lower left") corner
        of the grid cells.
    mltres : array
        width, in magnetic local time, of the cells with lower left corners described 
        by mlat and mlt. Notice that this width changes with latitude, while the 
        latitudinal width is fixed, determined by the `dr` parameter

    
    """

    r0 = dr * (2*K + 1)/2.

    assert M0 % (K + 1) == 0 # this must be fulfilled

    grid = {}

    M = M0
    grid[90 - r0 - dr] = np.linspace(0, 24 - 24./M, M) # these are the lower limits in MLT

    for i in range(N - 1):

        M = M *  (1 + 1./(K + i + 1.)) # this is the partion for i + 1

        grid[90 - (r0 + (i + 1)*dr) - dr] = np.linspace(0, 24 - 24./M, int(M)) # these are the lower limits in MLT
        # jreistad added int() to the M argument in np.linspace call in the above line. 
        # I think line 43 forces this to always be the case. (2020-06-18)

    mlats = []
    mlts = []
    mltres = []
    for key in sorted(grid.keys()):
        mltres_ = sorted(grid[key])[1] - sorted(grid[key])[0]
        for mlt in sorted(grid[key]):
            mlats.append(key)
            mlts.append(mlt)
            mltres.append(mltres_)

    return np.array(mlats), np.array(mlts), np.array(mltres)


def sdarngrid(dlat = 2, dlon = 2, latmin = 58, return_mltres = False):
    """ generate same grid as used by Matthias Forster in EDI study

        The grid is almost equal area but not exactly.

        Translated from MF's IDL code

    2020.03.06 (SMH): Added return_mltres keyword to make output of this function parallel to output of equal_area_grid
    """
    DEBUG = False

    nlats = int((90. - latmin)/dlat)
    mlats = latmin + np.arange(nlats, dtype = np.float32)*dlat # lower latitude corner

    npixel = 0

    for i in range(nlats):
        circle = 360. * np.cos(mlats[i] * d2r + dlat/2*d2r)
        nlongs = int(np.round( circle / dlon ))
        dellon = 360. / np.float32(nlongs )
        npixel = npixel + nlongs
        mlat   = np.zeros(nlongs) + mlats[i]
        mlon   = np.arange(nlongs, dtype = np.float32) * dellon
        if i == 0:
            mlat_arr = np.array([mlat]).flatten()
            mlon_arr = np.array([mlon]).flatten()
        else:
            mlat_arr = np.hstack((mlat_arr, np.array(mlat)))
            mlon_arr = np.hstack((mlon_arr, np.array(mlon)))


    grid = np.vstack((mlat_arr, mlon_arr)) # mlat/mlon grid pattern
    MLT_arr = mlon_arr/15. % 24.         # [hrs]

    if return_mltres:
        # mltzeros = np.where(MLT_arr == 0)[0]
        mltzeros = np.where(np.isclose(MLT_arr,0))[0]
        if mltzeros[-1] == (MLT_arr.size-1):
            mltzeros = mltzeros[:-1]
        stopmlt = np.append(mltzeros[1:],MLT_arr.shape[0])
        mltres = np.zeros(MLT_arr.shape)

        if DEBUG:
            print("{:10s}, {:10s}, {:10s}".format("startmlt","stopmlt","startmlat"))
        for istart in range(len(mltzeros)):
            if DEBUG:
                print("{:10.2f}, {:10.2f}, {:10.2f}".
                      format(MLT_arr[mltzeros[istart]],
                             MLT_arr[mltzeros[istart]+1],
                             forstermlat[mltzeros[istart]]))

            mltres[mltzeros[istart]:stopmlt[istart]] = MLT_arr[mltzeros[istart]+1]-MLT_arr[mltzeros[istart]]


        return grid, MLT_arr, mltres
    else:
        return grid, MLT_arr


def bin_number_OLD(grid, mlat, mlon, mlt = True):
    """ Return bin number for data points at given mlat, mlon.

        The grid should be a 2 x N array, with mlats and mlons, respectively

        mlat is the mlat coordinate of inputs
        mlon is the mlon coordinate of inputs

        if mlt is True, mlon is interpreted as MLT

        if mlat and mlon are arrays of size M, the resulting
        array will be of size M, containing integers between 0 and N - 1
        The number indicates the index of the grid cell in which the data point should be placed

        Translated from IDL code by Matthias Forster
    """

    mlat = mlat.flatten()
    mlon = mlon.flatten()

    assert grid.shape[0] == 2
    assert len(mlat) == len(mlon)

    res = np.zeros(len(mlat), dtype = np.int16) - 1

    # check if given mlon is really mlt, and convert to [-180, 180]
    if mlt: mlon = mlon * 15

    grid[1] = (grid[1] + 360) % 360 # normalize grid mlon to [0, 360]
    mlon    = (mlon    + 360) % 360 # same with mlon

    unique_mlats = np.sort(np.unique(grid[0]))
    dlat         = np.median(unique_mlats[1:] - unique_mlats[:-1])
    latmin       = np.min(unique_mlats)

    for i in range(len(mlat)):
        if (mlat[i] >= latmin) & (mlat[i] <= 90.):
            mlatbin = np.searchsorted(unique_mlats, mlat[i], side = 'right')-1
            if mlatbin >= len(unique_mlats):
                continue
            iii     = np.where( (grid[0] == unique_mlats[mlatbin]) )[0]
            ilon    = np.argmin(np.abs(grid[1][iii] - mlon[i]))
            res[i]  = iii[ilon]
    
    return res


def bin_number(grid,mlat,mlon,
               mlt=True,
               verbose=False):
    """ Return bin number for data points at given mlat, mlon.

        The grid should be a 2 x N array, with mlats and mlons, respectively

        mlat is the mlat coordinate of inputs
        mlon is the mlon coordinate of inputs

        if mlt is True, mlon is interpreted as MLT

        if mlat and mlon are arrays of size M, the resulting
        array will be of size M, containing integers between 0 and N - 1
        The number indicates the index of the grid cell in which the data point should be placed

    I (SMH) am confident that the original routine bin_number above does something wrong around the 24/0 MLT line.
    SMH 2020/04/01
    """
    mlat = mlat.flatten()
    mlon = mlon.flatten()
    
    assert grid.shape[0] == 2
    assert len(mlat) == len(mlon)
    
    res = np.zeros(mlat.size,dtype=np.int64)-1

    if mlt: mlon = mlon * 15
    grid[1] = (grid[1] + 360) % 360 # normalize grid mlon to [0, 360]
    mlon    = (mlon    + 360) % 360 # same with mlon

    dlat         = np.median(np.sort(np.unique(grid[0]))[1:] - np.sort(np.unique(grid[0]))[:-1])
    mlonres      = get_sdarngrid_mlon_resolution(grid)

    for ibin,(tmpmlon,tmpdmlon,tmpmlat) in enumerate(zip(grid[1], mlonres,grid[0])):
        
        tmpi = np.where((mlat >= tmpmlat) & (mlat < (tmpmlat+dlat+1e-4)) & (mlon >= tmpmlon) & (mlon < (tmpmlon+tmpdmlon+1e-4)))[0]
        if verbose:
            print(f"{ibin:02d}, {tmpmlat:5.2f}, {tmpmlat+dlat:5.2f}, {tmpmlon:5.2f}, {tmpmlon+tmpdmlon:5.2f}, {tmpi.size:05d}")
        res[tmpi] = ibin

    return res


def get_sdarngrid_mlon_resolution(grid):
    """
    Get the mlon resolution of sdarngrid (surprisingly not entirely trivial)

    # EXAMPLE
    forsterdmlat = 2
    forsterdmlon = 2
    forsterminlat = 70

    grid, forstermlt, forsterdmlt = sdarngrid(dlat=forsterdmlat,dlon=forsterdmlon,latmin=forsterminlat,return_mltres = True)
    mlonres = get_sdarngrid_mlon_resolution(grid)
    # Voilá

    SMH 2020/04/02
    """
    DEBUG = False

    mlonzeros = np.where(grid[1] == 0)[0]
    if mlonzeros[-1] == (grid[1].size-1):
        mlonzeros = mlonzeros[:-1]
    stopmlt = np.append(mlonzeros[1:],grid[1].shape[0])
    mlonres = np.zeros(grid[1].shape)
    
    if DEBUG:
        print("{:10s}, {:10s}, {:10s}".format("startmlt","stopmlt","startmlat"))
    for istart in range(len(mlonzeros)):
        if DEBUG:
            print("{:10.2f}, {:10.2f}, {:10.2f}".
                  format(grid[1][mlonzeros[istart]],
                         grid[1][mlonzeros[istart]+1],
                         forstermlat[mlonzeros[istart]]))
    
        mlonres[mlonzeros[istart]:stopmlt[istart]] = grid[1][mlonzeros[istart]+1]-grid[1][mlonzeros[istart]]

    return mlonres
