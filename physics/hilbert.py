import numpy as np
from scipy import sparse as ssp
import scipy.special
import numba
import time
import os
import pdb

@numba.njit(numba.float64[:](numba.float64[:],
                             numba.int64,
                             numba.int64,
                             numba.int64,
                             numba.int8[:,:],
                             numba.int64[:],
                             numba.float64[:,:],
                             numba.types.DictType(numba.int64, numba.int64)),
            parallel=True, fastmath=True)
def getMatVec(v, Nphi, Ne, NH, hilb, hilbLen, T4, dictx):
    """
    Hamiltonian operator (shell matrix)
    about 300x slower than getMat, but saves tremendous memory
        
    inputs:
    -------
    v        : np.array (float of size NH)
    Nphi     : int, number of orbitals
    Ne       : int, number of electrons
    NH       : int, size of Hilbert space
    hilb     : np.array of ints, of NH rows and Ne columns with the 
               full Hilbert space
    hilbLen  : np.array of ints, size of Hilbert space in each momentum subsector
    T4       : np.array of size Nphi x Nphi with elements of the interaction tensor
    dictx    : hashmap: keys are decimal representation of electron occupancies,
                        e.g. Ne = [0, 1, 3, 6] = 1 + 2 + 8 + 64 = 75
                        values are array indices of the state in hilb
        
    outputs:
    --------
    vOut: np.array (float of size NH), result of Hv
    """
        
    vOut = np.zeros(NH)

    for cHilb in numba.prange(NH):
        eOcc = hilb[cHilb] # occupied electrons
            
        diagel = 0.0
            
        #find pairs
        for p1 in np.arange(Ne):
            for p2 in np.arange(p1+1, Ne):
                c1 = eOcc[p1] # orbital of first occupied electron
                c2 = eOcc[p2] # orbital of second occupied electron
    
                # diagonal terms
                diagel += -T4[(c1-c2)%Nphi, (c1-c2)%Nphi] + \
                               T4[(c2-c1)%Nphi, 0]
                                                   
                # find new pairs
                for cx in np.arange(1, Nphi):
                    c1new = (c1+cx)%Nphi
                    c2new = (c2-cx)%Nphi
                    if c1new >= c2new:
                        continue
                            
#                     if c1new in eOcc or c2new in eOcc:
#                         continue

                    ####################
#                     numba does not recognize contains (the 'in' keyword)
                    searchIn = False
                    for eOrb in eOcc:
                        if c1new == eOrb or c2new == eOrb:
                            searchIn = True
                            break
                        if searchIn:
                            continue
                    ####################
                        
                    else:
                        # at this point, we are sure that c2new > c1new,
                        # and that they are both not in eOcc

                        # populate eOccNew
#                         eOccNew = np.zeros(self.Ne, dtype='int8') # array of length Ne
                        eOccNew = np.zeros(Ne, dtype=numba.int64) # array of length Ne
                        cNeOld = 0
                        state = 0
                        for cNe in range(Ne):
                            while cNeOld == p1 or cNeOld == p2:
                                cNeOld += 1

                            if state == 0:
                                if cNeOld < Ne and eOcc[cNeOld] < c1new:
                                    eOccNew[cNe] = eOcc[cNeOld]
                                    cNeOld += 1
                                else:
                                    eOccNew[cNe] = c1new
                                    p1new = cNe
                                    state = 1

                            elif state == 1:
                                if cNeOld < Ne and eOcc[cNeOld] < c2new:
                                    eOccNew[cNe] = eOcc[cNeOld]
                                    cNeOld += 1
                                else:
                                    eOccNew[cNe] = c2new
                                    p2new = cNe
                                    state = 2

                            else:
                                eOccNew[cNe] = eOcc[cNeOld]
                                cNeOld += 1


                        # find index of eOccNew in hilb, again binary search
                        indNew = dictx[np.sum(2**eOccNew)] # self.indexOf(eOccNew)
                            
                        # (-1)^n = 1 - 2*(n%2)
                        #the four terms
                        matrixel = ((1 - 2*((p1new+p2new+p1+p2 + 1)%2)) * 
                                     T4[(c2new-c1new)%Nphi, (c2-c1new)%Nphi]) + \
                                   ((1 - 2*((p1new+p2new+p1+p2)%2)) * 
                                     T4[(c2new-c1new)%Nphi, (c1-c1new)%Nphi])
                            
                        # update vOut
                        vOut[cHilb] += matrixel * v[indNew]
        vOut[cHilb] += diagel * v[cHilb]
    return vOut

@numba.njit(numba.float64[:,:](numba.int64,
                               numba.int64,
                               numba.int64,
                               numba.int8[:,:],
                               numba.int64[:],
                               numba.float64[:,:],
                               numba.types.DictType(numba.int64, numba.int64),
                               numba.int64,
                               numba.int64),
            fastmath=True)
def getMat(Nphi, Ne, NH, hilb, hilbLen, T4, dictx, i=0, Nth=1):
    """
    Hamiltonian operator for sparse matrix creation in COO format

    inputs:
    -------
    v        : np.array (float of size NH)
    Nphi     : int, number of orbitals
    Ne       : int, number of electrons
    NH       : int, size of Hilbert space
    hilb     : np.array of ints, of NH rows and Ne columns with the 
               full Hilbert space
    hilbLen  : np.array of ints, size of Hilbert space in each momentum subsector
    T4       : np.array of size Nphi x Nphi with elements of the interaction tensor
    dictx    : hashmap: keys are decimal representation of electron occupancies,
                        e.g. Ne = [0, 1, 3, 6] = 1 + 2 + 8 + 64 = 75
                        values are array indices of the state in hilb
    i        : np.int, thread index
    Nth      : np.int, total number of threads
                        
    outputs:
    --------
    dij: np.array, dij[0, :] contains matrix elements
                   dij[1, :] contains row indices i
                   dij[2, :] contains col indices j
    """
    Nterms = int(NH * Ne * (Ne - 1) * (Nphi - Ne) / (4 * Nth))
    dij = np.zeros((3, Nterms))

#     for m in np.arange(100):
#         dij += 1.0
#     return dij

    cterm = 0

    NH_parr = np.arange((i*NH)//Nth, ((i+1)*NH)//Nth)
    
    for cHilb in NH_parr:
        eOcc = hilb[cHilb] # occupied electrons

        diagel = 0.0

        #find pairs
        for p1 in np.arange(Ne):
            for p2 in np.arange(p1+1, Ne):
                c1 = eOcc[p1] # orbital of first occupied electron
                c2 = eOcc[p2] # orbital of second occupied electron

                # diagonal terms
                diagel += -T4[(c1-c2)%Nphi, (c1-c2)%Nphi] + \
                           T4[(c2-c1)%Nphi, 0]

                # find new pairs
                for cx in np.arange(1, Nphi):
                    c1new = (c1+cx)%Nphi
                    c2new = (c2-cx)%Nphi
                    if c1new >= c2new:
                        continue

#                         if c1new in eOcc or c2new in eOcc:
#                             continue

                    ####################
#                         numba does not recognize contains (the 'in' keyword)
                    searchIn = False
                    for eOrb in eOcc:
                        if c1new == eOrb or c2new == eOrb:
                            searchIn = True
                            break
                    if searchIn:
                        continue
                    ####################

                    else:
                        # at this point, we are sure that c2new > c1new,
                        # and that they are both not in eOcc

                        # populate eOccNew
#                             eOccNew = np.zeros(self.Ne, dtype='int8') # array of length Ne
                        eOccNew = np.zeros(Ne, dtype=numba.int64) # array of length Ne
                        cNeOld = 0
                        state = 0
                        for cNe in range(Ne):
                            while cNeOld == p1 or cNeOld == p2:
                                cNeOld += 1

                            if state == 0:
                                if cNeOld < Ne and eOcc[cNeOld] < c1new:
                                    eOccNew[cNe] = eOcc[cNeOld]
                                    cNeOld += 1
                                else:
                                    eOccNew[cNe] = c1new
                                    p1new = cNe
                                    state = 1

                            elif state == 1:
                                if cNeOld < Ne and eOcc[cNeOld] < c2new:
                                    eOccNew[cNe] = eOcc[cNeOld]
                                    cNeOld += 1
                                else:
                                    eOccNew[cNe] = c2new
                                    p2new = cNe
                                    state = 2

                            else:
                                eOccNew[cNe] = eOcc[cNeOld]
                                cNeOld += 1


                        # find index of eOccNew in hilb, again binary search
                        indNew = dictx[np.sum(2**eOccNew)] # self.indexOf(eOccNew)

                        # (-1)^n = 1 - 2*(n%2)
                        #the four terms
                        matrixel = ((1 - 2*((p1new+p2new+p1+p2 + 1)%2)) * 
                                     T4[(c2new-c1new)%Nphi, (c2-c1new)%Nphi]) + \
                                   ((1 - 2*((p1new+p2new+p1+p2)%2)) * 
                                     T4[(c2new-c1new)%Nphi, (c1-c1new)%Nphi])

                        # update vOut
                        dij[0, cterm] = matrixel
                        dij[1, cterm] = indNew
                        dij[2, cterm] = cHilb
                        cterm += 1

        dij[0, cterm] = diagel
        dij[1, cterm] = cHilb
        dij[2, cterm] = cHilb
        cterm += 1
    return dij
    
@numba.njit(numba.types.DictType(numba.int64, numba.int64)
            (numba.int8[:,:]))
def getDict(hilb):
    """
    get a hash table mapping keys (state binary string) to indices
    
    inputs:
    --------
    hilb: numba.int8[:,:]
    
    
    outputs:
    --------
    dict: keys - decimal representation of basis state
          e.g. [0, 1, 3, 7] -> 2**0 + 2**1 + 2**3 + 2**7 = 139
          values - index 
    """
    keys = np.sum(2**hilb, axis=1)
    d = numba.typed.Dict.empty(
        key_type=numba.int64,
        value_type=numba.int64,
    )

    for i in range(keys.size):
        d[keys[i]] = i

    return d


@numba.njit(#numba.types.ListType(numba.float64[:,:])(numba.int64,
            numba.float64[:,:,:](numba.int64,
                                 numba.int64,
                                 numba.int64,
                                 numba.int8[:,:],
                                 numba.int64[:],
                                 numba.float64[:,:],
                                 numba.types.DictType(numba.int64, numba.int64),
                                 numba.int64),
            parallel=True, fastmath=True)
def getMatAux(Nphi, Ne, NH, hilb, hilbLen, T4, dictx, Nth=1):
    """
    auxiliary function to enable parallelization of the matrix
    see getMat for documentation
    """
    Nterms = int(NH * Ne * (Ne - 1) * (Nphi - Ne) / (4 * Nth))
    dij = np.zeros((Nth, 3, Nterms))
#     dij = numba.typed.List()
    
    for i in numba.prange(Nth):
        dij[i, :, :] = getMat(Nphi, Ne, NH, hilb, hilbLen, T4, dictx, i, Nth)
#         dij.append(getMat(Nphi, Ne, NH, hilb, hilbLen, T4, dictx, i, Nth))
        
    return dij


def dijToCsr(dij, NH):
    """
    convert dij to scipy sparse matrix
    """
    H = scipy.sparse.coo_matrix((NH, NH))
    
    Nth = len(dij)
    
    for i in np.arange(Nth):
        H += scipy.sparse.coo_matrix((dij[i, 0, :], 
                                     (dij[i, 1, :], dij[i, 2, :])),
                                     shape=(NH, NH))
#         H += scipy.sparse.coo_matrix((dij[i][0, :], 
#                                      (dij[i][1, :], dij[i][2, :])),
#                                      shape=(NH, NH))
        
    H = H.tocsr()
        
    return H