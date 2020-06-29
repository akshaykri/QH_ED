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

class Torus:
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
            mmax = int(np.round(3*np.maximum(self.Lx, self.Ly)))
        ms = np.r_[-mmax:mmax+1]
        return ((2*np.pi/self.Lx) * ms,
                (2*np.pi/self.Ly) * ms, 
                mmax)
        
        
class Potential:
    """
    Encapsulates all useful information about a potential.
    """
    
    def __init__(self, torus, hamParams, vParams):
        """
        torus     : instance of Torus class
        hamParams : dict with keys 'alpha', 'Nphi', and 'n'
        vParams   : dict with optional keys 'power', 'haldane' etc.  
        """
        self.torus = torus
        self.hamParams = hamParams
        self.vParams = vParams
        self.V1 = None
        self.V2 = None # two particle
    
    @staticmethod
    def makeFF(kx_arr, ky_arr, alpha=1.0, n=0, **kwargs):
        """
        make the single particle form factor
        In the n = 0 LLL,
        FF(kx, ky) = exp( -(l_B^2/4) (k_x^2 + k_y^2) ) 
                   * exp(i/2 k_x k_y)
        In higher LL's, use the Laguerre polynomial
        (cf. Jainendra Jain, composite Fermions, Eq. 3.205, 
         and Kun Yang, PRB 88 241105(R), 2013, Eq. 8)
        
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
        
        absFF = np.outer(np.exp(-0.25 * alpha * kx_arr**2), np.exp(-0.25 * ky_arr**2 / alpha)) * \
                scipy.special.eval_laguerre(n, 0.5 * (alpha * kx_arr[:, np.newaxis]**2 + 
                                                      ky_arr**2 / alpha))
            
        phaseFF = np.exp(-0.5j * np.outer(kx_arr, ky_arr))
        
        return absFF * phaseFF
    
    @staticmethod
    def vPower(k, n=1, x=np.inf, l=1000):
        """
        obtain V(k) for a power-law interaction

        inputs:
        -------
        k     : np.ndarray, absolute value of momentum (2-d array)
        n     : float, power law fall-off
        x     : float, Gaussian envelope length scale
        l     : int, number of divisions in integration quadrature

        outputs:
        --------
        np.ndarray: V(k) at continuum k 
        """

        # if only Gaussian
        if n == 0:
            Vk = 2 * np.pi * x**2 * np.exp(-x**2 * k**2 / 2)
            Vk[np.abs(k) < 1e-8] = 0
            return Vk
        else:
            # if Coulomb
            if n == 1 and x is np.inf:
                Vk = np.zeros_like(k)
                Vk[np.abs(k) > 1e-8] = 2*np.pi / k[np.abs(k) > 1e-8]
                return Vk
            else:
                def integrand(r, k1):
                    return r * scipy.special.jv(0, k1*r) * 1/r**n * np.exp(-0.5 * (r/x)**2)

                ret = np.zeros_like(k)
                Nx, Ny  = ret.shape

                for i in range(Nx):
                    for j in range(Ny):
                        val1, _ = scipy.integrate.quad(integrand, 0, 1, args=(k[i, j]), limit=l)
                        val2, _ = scipy.integrate.quad(integrand, 1, np.inf, args=(k[i, j]), limit=l)
                        ret[i, j] = 2 * np.pi * (val1 + val2)

                return ret
            
    @staticmethod
    def vPseudopot(k, Vm=None):
        """
        obtain V(k), given the Haldane pseudopotentials

        inputs:
        -------
        k     : np.ndarray, absolute value of momentum (2-d array)
        Vm    : np.ndarray, Haldane pseudopotentials, default to Coulomb in n=0 LL

        outputs:
        --------
        np.ndarray : V(k) at continuum k 
        """
        
        if Vm is None:
            Vm = np.zeros(20)
            for m in np.arange(20):
                ## see Yoshioka (4.39)
                Vm[m] = np.sqrt(np.pi)/2 * scipy.special.factorial(2*m) / (
                    2**(2*m) * scipy.special.factorial(m)**2)
                
        Vk = np.zeros_like(k)
        ## see Yoshioka, (4.38)
        for m in np.arange(len(Vm)):
            Vk += 2 * Vm[m] * scipy.special.eval_laguerre(m, k**2)
        
        
        Vk[np.abs(k) < 1e-8] = 0
        
        return Vk

    
    @staticmethod
    def vDelta(kx, ky, x0=0.0, y0=0.0):
        """
        obtain V(k), for V(r) = delta(r-r0)
        useful for computing the pair correlation function

        inputs:
        -------
        kx, ky: np.array, x and y components of momentum
        x0, y0: coordinates of position r0
        
        outputs:
        --------
        np.ndarray : V(kx, ky) at continuum k 
        """
        
        Vk = np.outer(np.exp(1j*kx*x0), np.exp(1j*ky*y0))
        
        return Vk
    
    @staticmethod
    def getVk(kx, ky, 
              vParams={'power': (1.0, {})}):
        """
        get the k-space potential
        
        inputs:
        -------
        kx, ky : np.array, x and y components of momentum
        vParams: dict with the following keys
                 'power' 
                 'haldane'
                 'delta'
                 the values are tuples (amplitude, dict), where the dict
                 specifies parameters of the interaction
        
        outputs:
        --------
        float : V(kx, ky) at continuum k 
        """
        
        k_abs = np.sqrt(kx[:, np.newaxis]**2 + ky**2)
        Vk = np.zeros((len(kx), len(ky)), dtype='complex128')
        
        if 'power' in vParams:
            Vk += vParams['power'][0] * Potential.vPower(k_abs, **vParams['power'][1])
        if 'haldane' in vParams:
            Vk += vParams['haldane'][0] * Potential.vPseudopot(k_abs, **vParams['haldane'][1])
        if 'delta' in vParams:
            Vk += vParams['delta'][0] * Potential.vDelta(kx, ky, **vParams['delta'][1])
        
        return Vk
    
    @staticmethod
    def applyFF(kx, ky, Vk, hamParams):
        """
        return the V(k) with form factor
        
        inputs:
        -------
        kx, ky  : np.array, x and y components of momentum
        Vk      : dict with the following keys
                   'power' 
                   'haldane'
                   'delta'
                   the values are tuples (amplitude, dict), where the dict
                   specifies parameters of the interaction
        hamParams: dict with keys 'Nphi', 'alpha', 'n'
        
        outputs:
        --------
        np.ndarray : V(kx, ky) at continuum k WITH form factor
        """
        
        FF = Potential.makeFF(kx, ky, **hamParams)
        return Vk * FF * FF / (2 * np.pi * hamParams['Nphi'])
        
    @staticmethod
    def getV2(torus, 
              vParams={}, 
              hamParams={}):
        """
        return a two dimensional np.array, with the 
        effective potential at q = (k_x, k_y),
        where k_x and k_y are given by the torus 
        
        inputs:
        -------
        torus     : Torus object
        vParams   : dict of dicts:
                    'power' with keys A, n, x, l
                    'haldane' with keys A, Vm
        Vk        : actual matrix with Vk values, default None
        hamParams : dict with keys alpha, n, Nphi
        
        outputs:
        --------
        np.ndarray: 
        """
        
        kx, ky, _ = torus.get_karr()
        VkFF = Potential.applyFF(kx, ky, 
                                 Potential.getVk(kx, ky, vParams), 
                                 hamParams)
        
        return VkFF
    
    def setV2(self):
        """
        set the V2 attribute (interaction)
        """
        self.V2 = self.getV2(self.torus, self.vParams, self.hamParams)