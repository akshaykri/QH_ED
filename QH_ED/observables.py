import numpy as np
from scipy import sparse as ssp
import scipy.special
import numba
import time
import os
import pdb
from functools import partial

from QH_ED.utils import Utils
from QH_ED.landau import Torus, Potential
import QH_ED.hilbert
import QH_ED.hilbert

class Observables:
    """
    class to handle observables
    """
    
    def __init__(self, torus, hamParams, hilbObj):
        """
        torus     : instance of Torus class
        hamParams : dict with keys 'alpha', 'n', 'Nphi'
        hilbObj   : insance of Hilbert class
        """
        self.torus = torus
        self.hamParams = hamParams
        self.hilbObj = hilbObj
    
    def gr(self, r_arr, V):
        """
        pair correlation function g(r) = 
        1/(\rho_0 N_e) \langle \sum_{i \neq j} \delta (r_i - r_j) \rangle
        
        inputs:
        ---------
        r_arr  : np.array, N x 2 array of floats, 
                 positions at which to calculate g(r)
        V      : np.array, eigenvector
        
        outputs:
        ---------
        g(r): g(r) over values of r in r_arr
        """

        kx, ky, mmax = self.torus.get_karr()
        Npoints = len(r_arr)
        Nth = 4
        
        grOut = np.zeros(Npoints)
        
        for cR in np.arange(Npoints):
            vDeltaParams = {'delta': (1.0, {'x0': r_arr[cR, 0], 'y0': r_arr[cR, 1]})}
            VkFF = Potential.applyFF(kx, ky, 
                             Potential.getVk(kx, ky, vDeltaParams), 
                             self.hamParams)


            T4_delta = Utils.make4tensorsym(self.torus, VkFF)

            newMatVec = partial(QH_ED.hilbert.getMatVecC, 
                                Nphi = self.hilbObj.Nphi, 
                                Ne = self.hilbObj.Ne, 
                                NH = self.hilbObj.NH, 
                                hilb = self.hilbObj.hilb, 
                                hilbLen = self.hilbObj.hilbLen, 
                                T4 = T4_delta,
                                dictx = self.hilbObj.dictx)

            elem = np.inner(V*(1.0+0.0j), newMatVec(V*(1.0+0.0j)))
            assert np.imag(elem) < 1e-12

            grOut[cR] = np.real(elem)
        
        denom = 2 * np.pi * self.hilbObj.Nphi / (self.hilbObj.Ne**2)
        
        return grOut * denom
    
    def grAux(self, nX, nY, V):
        """
        calculate the pair correlation function over a grid of equidistat points
        on the torus
        
        inputs:
        --------
        nX : int, number of lattice points along Lx
        nY : int, number of lattice points along Ly
        V  : np.ndarray, eigenvector
        
        outputs:
        gR : np.array of shape (nX, nY), with g(r) evaluated on the lattice
        
        """
        X_arr = 0.5 * np.arange(nX+1) * self.torus.Lx / nX
        Y_arr = 0.5 * np.arange(nY+1) * self.torus.Ly / nY
        
        X_temp, Y_temp = np.meshgrid(X_arr, Y_arr)
        r_arr = np.c_[X_temp.ravel(), Y_temp.ravel()]
        
        grOut = self.gr(r_arr, V)
        grOut = grOut.reshape(nY+1, nX+1).T
        
        return np.c_[np.r_[grOut, grOut[-2::-1,:]], 
                     np.r_[grOut[:,-2::-1], grOut[-2::-1,-2::-1]]]