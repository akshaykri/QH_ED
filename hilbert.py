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
    
    

class Hilbert:
    
    def __init__(self, Nphi, Ne):
        self.Nphi = Nphi
        self.Ne = Ne
        
    def getMatVec(self, v):
        # hilbs must be a list of arrays, corresponding to all the sub-sectors in the
        # Hilbert space
        NH = len(hilb) # size of Hilbert space

        vOut = np.zeros(NH)

        for cHilb in range(NH):
            eOcc = hilb[cHilb] # occupied electrons

            #find pairs
            for p1 in range(Ne):
                for p2 in range(p1+1, Ne):
                    c1 = eOcc[p1] # orbital of first occupied electron
                    c2 = eOcc[p2] # orbital of second occupied electron

                        # find new pairs
                    for cx in range(1, Nphi):
                        c1new = (c1+cx)%Nphi
                        c2new = (c2-cx)%Nphi
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
                            for cNe in range(Ne):
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
                            indNew = indexOf(eOccNew, hilb)


                            matrixel = 0
                            # add the four terms, with correct signs

                            # update vOut
                            vOut[indNew] += matrixel
        return vOut

class Torus(object):
    """
    Encapsulates info about the geometry of the torus
    """

    def __init__(self, Nphi, aspect_ratio = 1., theta_x = 0., theta_y = 0., params={}):
        self.Nphi = Nphi
        self.aspect_ratio = aspect_ratio
        self.Lx = np.sqrt(2*np.pi*Nphi/aspect_ratio)
        self.Ly = np.sqrt(2*np.pi*Nphi*aspect_ratio)
        self.theta_x = theta_x
        self.theta_y = theta_y
        
    def get_karr(self, mmax=None):
        """
        inputs:
        --------
        mmax     : int, maximum Fourier harmonic
        
        outputs:
        --------
        np.array : one dimensional array 
                   [-2 \pi M / Lx, ..., -2 \pi / Lx, 0, 2 \pi / Lx, ..., 2 \pi M / Lx]
        np.array : one dimensional array 
                   [-2 \pi M / Ly, ..., -2 \pi / Ly, 0, 2 \pi / Ly, ..., 2 \pi M / Ly]
        int      : M
        """
        if mmax is None:
            mmax = 3*np.max(self.Lx, self.Ly)
        ms = np.r_[-mmax:mmax+1]
        return ((2*np.pi/self.Lx) * ms,
                (2*np.pi/self.Ly) * ms, 
                mmax)
        
        
class Potential:
    """
    Encapsulates all useful information about a potential.
    """
    def __init__(self):
        self.V1 = None
        self.V2 = None # two particle
    
    def makeFF(self, kx_arr, ky_arr, alpha=1.0, n=0, **kwargs):
        """
        make the single particle form factor
        In the n = 0 LLL,
        FF(kx, ky) = exp( -(l_B^2/4) (k_x^2 + k_y^2) ) 
                   * exp(i/2 k_x k_y)
        
        inputs:
        -------
        kx_arr: array of kx values
        ky_arr: array of ky values
        alpha : float, mass anisotropy
        n     : int, LL index
        
        
        returns:
        --------
        np.ndarray: Form factor
        """
        
        if n == 0:
            absFF = np.outer(np.exp(-0.25 * alpha * kx_arr**2), np.exp(-0.25 * kys**2 / alpha))
            phaseFF = np.exp(-0.5j * np.outer(kx_arr, ky_arr))
        
        return absFF * phaseFF
    
    def VFourier(k, n=1, x=np.inf, l=1000, **kwargs):
        """
        obtain V(k)

        inputs:
        -------
        k     : np.ndarray, absolute value of momentum
        n     : float, power law fall-off
        x     : float, Gaussian envelope length scale
        l     : int, number of divisions in integration quadrature

        outputs:
        --------
        float : V(k) at continuum k 
        """

        # if only Gaussian
        if n == 0:
            Vk = 2 * np.pi * x**2 * np.exp(-x**2 * k**2)
            Vk[np.abs(k) < 1e-8] = 0
            return Vk
        else:
            # if Coulomb
            if n == 1 and x is np.inf:
                Vk = np.zeros_like(k_arr)
                Vk[np.abs(k) > 1e-8] = 2*np.pi / k[np.abs(k) > 1e-8]
            else:
                def integrand(r, k1):
                    return r * scipy.special.jv(0, k1*r) * 1/r**n * np.exp(-0.5 * (r/x)**2)

                ret = np.zeros_like(k_arr)
                Nx, Ny  = ret.shape

                for i in range(Nx):
                    for j in range(Ny):
                        val1, _ = scipy.integrate.quad(integrand, 0, 1, args=(k_arr[i, j]), limit=l)
                        val2, _ = scipy.integrate.quad(integrand, 1, np.inf, args=(k_arr[i, j]), limit=l)
                        ret[i, j] = 2 * np.pi * (val1 + val2)

                return ret
    
    def setV2(self, torus, vParams={}, hamParams={}):
        """
        set the V2 attribute (interaction)
        self.V2 is a two dimensional np.array, with the 
        effective potential at q = (k_x, k_y),
        where k_x and k_y are given by the torus 
        
        inputs:
        -------
        torus     : Torus object
        vParams   : dict with keys n, x, l
        hamParams : dict with keys alpha, n, Nphi
        
        outputs:
        --------
        None
        """
        
        kx, ky, _ = torus.get_karr()
        k_abs = np.sqrt(kx[:, np.newaxis]**2 + ky**2)
        
        Vk = self.VFourier(k_abs, **vParams)
        FF = self.makeFF(kx_arr, ky_arr, **hamParams)
        
        self.V2 = Vk * FF * FF / hamParams['Nphi']
        
        
    