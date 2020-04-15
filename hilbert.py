import numpy as np
from scipy import sparse as ssp
import scipy.special
import numba
import time
import os
import pdb


class Hilbert:
    """
    bare-bones jit class with matvec
    """
    
    def __init__(self, Nphi, Ne, sector, hilb, hilbLen, T4):
        """
        Nphi   : int, number of orbitals
        Ne     : int, number of electrons
        sector : int, momentum sector
        hilb   : np.array, NH x Ne of dtype int8, has the entire Hilbert space 
        hilbLen: size of Hilbert space in each subsector
        T4     : tensor of interaction elements V_{n1 n2 n3 n4} = V_{n1 0 n3 n1-n3}
        """
        
        self.Nphi = Nphi
        self.Ne = Ne
        self.sector = sector
        self.hilb = hilb
        self.hilbLen = hilbLen
        self.NH = len(self.hilb) # size of Hilbert space
        self.T4 = T4 
        
    def getMatVec(self, v):
        """
        Hamiltonian operator
        """
        vOut = np.zeros(self.NH)

        for cHilb in range(self.NH):
            eOcc = self.hilb[cHilb] # occupied electrons

            #find pairs
            for p1 in range(self.Ne):
                for p2 in range(p1+1, self.Ne):
                    c1 = eOcc[p1] # orbital of first occupied electron
                    c2 = eOcc[p2] # orbital of second occupied electron

                    # find new pairs
                    for cx in range(1, self.Nphi):
                        c1new = (c1+cx)%self.Nphi
                        c2new = (c2-cx)%self.Nphi
                        if c1new >= c2new:
                            continue
                        if c1new in eOcc or c2new in eOcc:
                            continue
                        else:
                            # at this point, we are sure that c2new > c1new,
                            # and that they are both not in eOcc

                            # populate eOccNew
                            eOccNew = np.zeros(self.Ne, dtype=numba.int8) # array of length Ne
                            cNeOld = 0
                            state = 0
                            for cNe in range(self.Ne):
                                if cNeOld == p1 or cNeOld == p2:
                                    cNeOld += 1

                                if state == 0:
                                    if eOcc[cNeOld] < c1new:
                                        eOccNew[cNe] = eOcc[cNeOld]
                                        cNeOld += 1
                                    else:
                                        eOccNew[cNe] = c1new
                                        p1new = cNe
                                        state = 1

                                elif state == 1:
                                    if eOcc[cNeOld] < c2new:
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
                            indNew = self.indexOf(eOccNew)

                            # (-1)^n = 1 - 2*(n%2)
                            #the four terms
                            matrixel = ((1 - 2*((p1new+p2new+p1+p2 + 1)%2)) * 
                                        (self.T4[(c1new-c2new)%self.Nphi, (c1-c2new)%self.Nphi] + \
                                         self.T4[(c2new-c1new)%self.Nphi, (c2-c1new)%self.Nphi])) + \
                                       ((1 - 2*((p1new+p2new+p1+p2)%2)) * 
                                        (self.T4[(c1new-c2new)%self.Nphi, (c2-c2new)%self.Nphi] + \
                                         self.T4[(c2new-c1new)%self.Nphi, (c1-c1new)%self.Nphi]))
                            
                            # update vOut
                            vOut[indNew] += matrixel
        return vOut
    
    
    def indexOf(self, state):#, l=0, r=-1):
        """
        return the index of a many-body state among an array
        of such states via a binary search
        
        """
        
        factor = np.sum(state) // self.Nphi 
        
        # assert np.sum(state) == np.sum(allStatesSector[0])
        if factor == 0:
            l = 0
        else:
            l = np.cumsum(self.hilbLen)[factor-1]
        
        r = np.cumsum(self.hilbLen)[factor]

        #binary search
        while r-l > 1:
            mid = (l+r)//2
            if self.isGreater(allStatesSector[mid], state):
                r = mid
            else:
                l = mid

        return l

    def isGreater(self, s1, s2):
        """
        return True if s1 comes 'lexicographically' after s2
        s1 and s2 must be of same length, and each must be sorted in
        ascending order
        comparison is done starting from the 'highest occupied orbital'
        e.g. [0, 2, 4, 8] comes after [1, 3, 5, 7]
        
        inputs:
        -------
        s1   : numpy array of ints
        s2   : numpy array of ints
        
        outputs:
        --------
        bool : True or false
        """
#         assert len(s1) == len(s2)
        N = len(s1)
        arr = range(N)
        for c in arr:
            if s1[N-c-1] > s2[N-c-1]:
                return True
            elif s1[N-c-1] < s2[N-c-1]:
                return False
        return False