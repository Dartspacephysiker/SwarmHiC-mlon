""" this module contains functions which can be used to speed up the calculation of G.T.dot(G) and G.T.dot(d), using BLAS functions, 
    and exploiting the block structure of G.

    G is assumed to be of the form (w1G0, w2G0, w3G0, ...) where w1, w2, w3... are weight matrices

    weighted_GTd_GTG:          Computes the lower triangular portions of each block in GTG, as well as the GTd vector
                               The blocks are organized in a matrix with the shape of GTG.
    fill_in_missing_triangles: Expands the result of of weighted_GTd_GTG, except the first row (which is GTd)

    weighted_GTd_GTG_array:    Same as weighted_GTd_GTG, but the unique elements, and GTd are now stored in one long array. This should use
                               less memory, but it is possibly a little slower.
    expand_GTG_and_GTd:        Expands the result of weighted_GTd_GTG_array.

    The idea is to use these functions with dask's map_blocks (iterating over G, and the weights), and then sum the results in the end.
    The result could then be expanded.

    KML Feb 2016 
"""

import numpy as np
import dask.array as da
from scipy.linalg.blas import csyrk


def weighted_GTd_GTG(G0, B0, weights):
    """
    return lower triangular portions of G.T.dot(G), where G has the following blocks:

    G = (diag(weights[0]).dot(G0)  diag(weights[1]).dot(G0)  ...)

    where weights are arrays with as many elements as there are rows in G0. Call this number N

    If G0 has M columns, and there are K weights, the returned array will have shape: (M*K, M*K)

    weights must be a (K, N) array
    """

    weights = weights.T
    K = weights.shape[0]
    M = G0.shape[1]

    # calculate the transpose
    G0T = G0.T  # shape (M, N)


    """ build a dict with block matrices: """
    # block index    
    I, J = np.meshgrid(range(K), range(K))
    # lower triangle, including diagonal:
    iii = I >= J 
    IU = I[iii].flatten()
    JU = J[iii].flatten()


    """ place the lower triangular blocks in the full matrix """
    GTG = np.zeros((M*K, M*K))
    for i, j in zip(IU, JU):
        w = np.sqrt(weights[i] * weights[j] + 0j)
        Gw = G0T*w
        GTG_ = csyrk(1, Gw, lower = 1).real # low-level BLAS function to compute G.T dot G - computes only lower triangular portion
        GTG[i*M:(i+1)*M, j*M:(j+1)*M] = GTG_

    GTd = np.zeros((M*K, 1  ))
    for i in range(K):
        GTd[i*M:(i+1)*M, :] = (G0T*weights[i]).dot( B0 )

    return np.vstack((GTd.flatten(), GTG))


def weighted_GTd_GTG_array(G0, B0, weights):
    """
    same as weighted_GTd_GTG, except that only the unique elements of GTG are kept
    They are stored in a long array (the first elements are GTd)
    The array can be expanded with expand_GTG_GTd

    """
    weights = weights.T
    K = weights.shape[0]
    M = G0.shape[1]

    # calculate the transpose
    G0T = G0.T  # shape (M, N)

    # indices of lower triangular portion
    tril_iii = np.tril_indices(M) # indices of lower triangular portion of blocks
    N = len(tril_iii[0]) # number of elements in lower triangular portion



    """ build a dict with block matrices: """
    # block index    
    I, J = np.meshgrid(range(K), range(K))
    # lower triangle, including diagonal:
    iii = I >= J 
    IU = I[iii].flatten()
    JU = J[iii].flatten()


    """ place the lower triangular blocks in the full matrix """
    GTG = np.zeros(N * len(IU))
    count = 0
    for i, j in zip(IU, JU):
        w = np.sqrt(weights[i] * weights[j] + 0j) # add complex number to allow for negative weights
        Gw = G0T*w
        GTG_ = csyrk(1, Gw, lower = 1).real # low-level BLAS function to compute G.T dot G - computes only lower triangular portion
        GTG[count * N:(count+1)*N] = GTG_[tril_iii[0], tril_iii[1]] # store only unique elements
        count += 1

    GTd = np.zeros((M*K, 1  ))
    for i in range(K):
        GTd[i*M:(i+1)*M, :] = (G0T*weights[i]).dot( B0 )

    return np.hstack((GTd.flatten(), GTG))[np.newaxis, :]


def fill_in_missing_triangles(GTG, K):
    """ Expand the result of weighted_GTd_GTG

        GTG is the matrix which is returned from weighted_GTG_and_GTd, apart from the top row. 
        That matrix has only the lower triangular portions filled in - the rest are zero.
        This function fills in the rest

        K is the number of weights - the number of square chunks in both directions
    """
    M = GTG.shape[0]//K # chunk size

    I, J = np.meshgrid(range(K), range(K))
    iii = I >= J 
    IU = I[iii].flatten()
    JU = J[iii].flatten()
    # make a lower triangular matrix:
    for i, j in zip(IU, JU):
        if i == j: # do not do the blocks on diagonal yet
            continue 
        GTG_ = GTG[i*M:(i+1)*M, j*M:(j+1)*M]
        GTG[i*M:(i+1)*M, j*M:(j+1)*M] += (GTG_.T - np.diag(GTG_.diagonal()))

    # now fill in the full matrix:
    GTG += (GTG.T - np.diag(GTG.diagonal()))
    return GTG


def expand_GTG_and_GTd(GTG_GTd, nweights, neq):
    """
    expand output from weighted_GTd_GTG_array into full GTG matrix (neq, neq) and GTd (neq, 1)
    """
    M = neq
    GTd       = GTG_GTd[:M*nweights ] # first part is GTd
    GTG_array = GTG_GTd[ M*nweights:] # second is GTG blocks
    GTGs = np.split(GTG_array, nweights*(nweights+1)/2) # split in number of blocks

    tril_iii = np.tril_indices(M) # indices of lower triangular portion of blocks

    GTG = np.zeros((nweights*neq, nweights*neq), dtype = np.float64)

    I, J = np.meshgrid(range(nweights), range(nweights))
    iii = I >= J 
    IU = I[iii].flatten()
    JU = J[iii].flatten()
    count = 0
    # make a lower triangular matrix:
    for i, j in zip(IU, JU):
        GTG_ = np.zeros((neq, neq))
        GTG_[tril_iii[0], tril_iii[1]] = GTGs[count]
        count += 1
        if i == j: 
            GTG[i*M:(i+1)*M, j*M:(j+1)*M] = GTG_
            continue # don't do the upper triangular portion of blocks on diagonal yet
        GTG_ += (GTG_.T - np.diag(GTG_.diagonal()))
        GTG[i*M:(i+1)*M, j*M:(j+1)*M] = GTG_

    # now fill in the full matrix:
    GTG += (GTG.T - np.diag(GTG.diagonal()))

    return GTd, GTG


if __name__ == '__main__':
    import time

    # Dimensions of test arrays:
    ###########################
    NWEIGHTS = 10 # number of weights
    DMULTIPLIER = 20 # number of data points divided by number of equations
    NEQ = 50 # number of equations

    # define random arrays for G0, weights, and B0 (d)
    ##################################################
    G0 = da.random.random((NEQ*NWEIGHTS*DMULTIPLIER,NEQ), chunks = (NEQ*NWEIGHTS, NEQ))
    weights = da.random.random((NWEIGHTS, G0.shape[0]), chunks = (NWEIGHTS, G0.chunks[0]))
    B0 = da.random.random((G0.shape[0], 1), chunks = (G0.chunks[0], 1))
    
    # Standard way (but using dask):
    ################################
    t = time.time()
    G = da.hstack(tuple(G0*(weights[i][:, np.newaxis]) for i in range(weights.shape[0])))    
    GTG = G.T.dot(G).compute()
    GTd = G.T.dot(B0)
    GTd = GTd.compute()
    print( 'standard way took %s seconds' % (time.time() - t))
    
    # BLAS + blocks in full matrix
    ##############################
    t = time.time()
    GTd_GTG = da.map_blocks(weighted_GTd_GTG, G0, B0, weights.T, chunks = (NEQ*NWEIGHTS +1, NEQ*NWEIGHTS))
    CHUNKSIZE = NEQ*NWEIGHTS
    GTd_GTG_num = np.zeros((CHUNKSIZE+1, CHUNKSIZE))
    for i in range(GTd_GTG.numblocks[0]):
        GTd_GTG_num = GTd_GTG_num + GTd_GTG[i*(CHUNKSIZE + 1):(i+1)*(CHUNKSIZE + 1)].compute()
    
    myGTd = GTd_GTG_num[0 ] # first row is GTd
    myGTG = GTd_GTG_num[1:] # the rest is GTG
    
    myGTG = fill_in_missing_triangles(myGTG, NWEIGHTS)
    print( 'BLAS + blocks in full GTG took %s seconds' % (time.time() - t))
    
    # BLAS + blocks in long arrays:
    ###############################
    t = time.time()
    N_NUM = NEQ*(NEQ+1)//2*NWEIGHTS*(NWEIGHTS+1)//2 + NEQ*NWEIGHTS # number of unique elements in GTG and GTd
    GTd_GTG = da.map_blocks(weighted_GTd_GTG_array, G0, B0, weights.T, chunks = (1, N_NUM) )
    GTd_GTG_num = np.zeros(N_NUM)
    for i in range(GTd_GTG.numblocks[0]):
        GTd_GTG_num = GTd_GTG_num + GTd_GTG[i].compute()
    
    myGTd2, myGTG2 = expand_GTG_and_GTd(GTd_GTG_num, NWEIGHTS, NEQ)
    
    print( 'BLAS + blocks in arrays took %s seconds (it may not be much faster, but it should use less than half the memory)' % (time.time() - t))

    # make sure all techniques give the same result
    print( '\nDo all methods give the same results?')
    print( np.all(np.isclose(myGTG2, GTG)), np.all(np.isclose(myGTG, GTG)), np.all(np.isclose(GTd.flatten(), myGTd2.flatten())), np.all(np.isclose(myGTd.flatten(), GTd.flatten())))

