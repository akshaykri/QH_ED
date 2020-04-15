"""
Landau level gymnastics

class Torus

class Potential
"""
import numpy as np
from scipy import sparse as ssp
import scipy.special
import numba
import time
import os
import pdb

class Torus(object):
    """
    Encapsulates info about the geometry of the torus
    """

    def __init__(self, Nphi, aspect_ratio = 1., theta_x = 0., theta_y = 0., **kwargs):
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
            mmax = int(round(3*np.max(self.Lx, self.Ly)))
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
        
        self.V2 = Vk * FF * FF / (2 * np.pi * hamParams['Nphi'])
        