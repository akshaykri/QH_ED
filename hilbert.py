import numpy as np
from scipy import sparse as ssp
import scipy.special
import numba
import time
import os
import pdb


class Utils:
    
    def __init__(self, Ne, Nphi, torus, potential=None):
        self.Ne = Ne
        self.Nphi = Nphi
        self.torus = torus
        self.potential = potential
        
    @staticmethod
    def writeHilbert(nuinv, NeMax, fol='/tigress/ak20/QH/hilbert/'):
        """
        Utility function to find the entire Hilbert space 
        of Ne electrons in Nphi orbitals at fixed filling nuinv = Nphi/Ne, 
        for Ne = 1, ..., NeMax
        in all momentum sectors
        each state is represented by the Ne occupied orbitals,
        e.g., [0, 2, 6, 7, 13, 15] is one of the states for Nphi = 18, Ne = 6
        it is in momentum sector 0+2+6+7+13+15 = 43

        inputs:
        -------
        nuinv     : int, inverse filling
        NeMax     : int, number of electrons
        fol       : str, disk location to write to

        outputs:
        --------
        np.ndarray: type int8, with (Nphi \choose Ne) rows and Ne columns
        """
        NphiMax = NeMax * nuinv

        fil = lambda Nphi, Ne, sector: 'Nphi{0:d}_Ne{1:d}_sector{2:d}'.format(Nphi, Ne, sector)
        sect_min = lambda Ne: ((Ne-1)*Ne)//2
        sect_max = lambda Nphi, Ne: ((2*Nphi-Ne-1)*Ne)//2

        np.save(fol+fil(1, 1, 0), np.array([[0]], dtype='int8'))
        np.save(fol+fil(1, 0, 0), np.array([[]], dtype='int8'))

        for Nphi in range(2, NphiMax+1): # 1, 2, ..., Nphi
            if Nphi <= NphiMax - NeMax:
                np.save(fol+fil(Nphi, 0, 0), np.array([[]], dtype='int8'))
            for Ne in range(max(1, Nphi - NphiMax + NeMax), 
                            min(Nphi+1, NeMax+1)): # 1, ..., Nphi
                for sector in range(sect_min(Ne), sect_max(Nphi, Ne)+1):
                    if Nphi == NphiMax and not sector%Nphi < Nphi//nuinv:
                        continue
                    try:
                        upper = np.load(fol+fil(Nphi-1, Ne, sector)+'.npy')
                    except FileNotFoundError:
                        upper = np.zeros((0, Ne), dtype='int8')
                    try:
                        lower = np.load(fol+fil(Nphi-1, Ne-1, sector-Nphi+1)+'.npy')
                    except FileNotFoundError:
                        lower = np.zeros((0, Ne-1), dtype='int8')

                    allStates = np.r_[upper,
                                      np.c_[lower, np.full((lower.shape[0], 1), Nphi-1, 
                                                            dtype='int8')]]
                    np.save(fol+fil(Nphi, Ne, sector), allStates)

                # clean-up
                for sector in range(sect_min(Ne-1), sect_max(Nphi-1, Ne-1)+1):
                    if nuinv*(Ne-1) != Nphi-1 or sector%(Nphi-1) >= (Nphi-1)//nuinv:
                        try:
                            os.remove(fol+fil(Nphi-1, Ne-1, sector)+'.npy')
                        except FileNotFoundError:
                            pass

            if nuinv*Ne != Nphi-1:
                for sector in range(sect_min(Ne), sect_max(Nphi-1, Ne)+1):
                    try:
                        os.remove(fol+fil(Nphi-1, Ne, sector)+'.npy')
                    except FileNotFoundError:
                        pass
    
    @staticmethod
    @numba.njit
    def indexOf(state, allStatesSector, l=0, r=-1):
        """
        return the index of a many-body state among an array
        of such states via a binary search
        
        """
        
        N = len(allStatesSector)


        # assert np.sum(state) == np.sum(allStatesSector[0])

        # l = 0
        if r == -1:
            r = N

        #binary search
        while r-l > 1:
            mid = (l+r)//2
            if self.isGreater(allStatesSector[mid], state):
                r = mid
            else:
                l = mid

        return l

    @staticmethod
    @numba.njit
    def isGreater(s1, s2):
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
        assert len(s1) == len(s2)
        N = len(s1)
        arr = range(N)
        for c in arr:
            if s1[N-c-1] > s2[N-c-1]:
                return True
            elif s1[N-c-1] < s2[N-c-1]:
                return False
        return False
    
    
        
    @numba.njit(numba.int16[:](numba.int8[:], numba.int64))
    def get_sector_id(states, Nphi):
        """
        return the sector id,
        e.g. for states = [0, 2, 6, 7, 13, 15] and Nphi = 18, it is
        43 / 18 = 7
        """
        Nh = len(states)
        NphiArr = np.arange(Nphi)
        sectorArr = np.zeros(Nh, dtype='int16')
        for i in range(Nh):
            sectorArr[i] = np.sum(NphiArr[states[i]]) % Nphi
        return sectorArr
    
    
    def make4tensorsym(self, pot):
        """
        make the 4-index tensor V_{n1 n2 n3 n4} of the two-body potential
        Since only two independent indices are needed, we first construct an Nphi x Nphi matrix
        """
        
        kx, ky, mmax = self.torus.get_karr()

        PhaseMat = np.exp(+2j*np.pi / Nphi * np.outer(np.arange(-max_m, max_m+1), 
                          np.arange(Nphi)))
        
        Vmod = np.zeros((Nphi, 2*max_m + 1), dtype="complex128")

        for c1 in np.arange(2*mmax + 1):
            new_ind = (c1 - mmax) % self.Nphi
            Vmod[new_ind, :] += pot.V2[c1, :]

        self.T4 = np.dot(Vmod, PhaseMat).T 
    
    
    @staticmethod
    def getHilb(Nphi, Ne, sector, fol='/tigress/ak20/QH/hilbert/'):
        """
        get the entire Hilbert space in a particular sector
        
        inputs:
        -------
        Nphi   : int, number of orbitals,
        Ne     : int, number of electrons
        sector : int, momentuum sector (modulo Nphi)
        fol    : str, location of directory
        
        outputs:
        --------
        hilb   : np.array of dtype int8, with Ne columns, and NH rows 
                 where NH is the size of the Hilbert space
        """
        
        hilb = np.zeros((0, Ne), dtype='int8')
        hilbLen = np.zeros(Ne+1, dtype='int8')
        
        for s in range(Ne+1):
            try:
                fil = np.load(fol+'Nphi{0:d}_Ne{1:d}_sector{2:d}.npy'.format(
                              Nphi, Ne, sector + s*Nphi))
            except:
                pass
            hilb = np.r_[ret, fil]
            hilbLen[s] = len(fil)
            
        return hilb, hilbLen

class Hilbert:
    """
    bare-bones jit class with matvec
    """
    
    def __init__(self, Nphi, Ne, sector, hilb, hilbLen):
        """
        Nphi   : int, number of orbitals
        Ne     : int, number of electrons
        sector : int, momentum sector
        hilb   : np.array, NH x Ne of dtype int8, has the entire Hilbert space 
        hilbLen: size of Hilbert space in each subsector
        """
        
        self.Nphi = Nphi
        self.Ne = Ne
        self.sector = sector
        self.hilb = hilb
        self.hilbLen = hilbLen
        
        
    def getMatVec(self, v):
        """
        Hamiltonian operator
        """
        NH = len(self.hilb) # size of Hilbert space

        vOut = np.zeros(NH)

        for cHilb in range(NH):
            eOcc = hilb[cHilb] # occupied electrons

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
                            eOccNew = np.zeros(Ne, dtype='int8') # array of length Ne
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
                            indNew = self.indexOf(eOccNew, self.hilb)


                            matrixel = 0
                            # add the four terms, with correct signs

                            # update vOut
                            vOut[indNew] += matrixel
        return vOut
    
    
    def indexOf(self, state, l=0, r=-1):
        """
        return the index of a many-body state among an array
        of such states via a binary search
        
        """
        
        NH = len(self.hilb)


        # assert np.sum(state) == np.sum(allStatesSector[0])

        # l = 0
        if r == -1:
            r = N

        #binary search
        while r-l > 1:
            mid = (l+r)//2
            if self.isGreater(allStatesSector[mid], state):
                r = mid
            else:
                l = mid

        return l

    def isGreater(s1, s2):
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
        assert len(s1) == len(s2)
        N = len(s1)
        arr = range(N)
        for c in arr:
            if s1[N-c-1] > s2[N-c-1]:
                return True
            elif s1[N-c-1] < s2[N-c-1]:
                return False
        return False