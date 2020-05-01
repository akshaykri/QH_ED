import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import sys

import landau
import utils
import hilbert

if __name__ == "__main__":
    
    alp_arr = np.r_[1.0:4.01:0.1] # 31
    asp_arr = np.array([0.8, 1, 1.25]) # 3
    Ne_arr = np.array([6, 7, 8]) # 3
    # 0 - 278
    
    ind = int(sys.argv[1])
    
    Ne = Ne_arr[ind // 93]
    Nphi = 3*Ne
    aspect_ratio = asp_arr[(ind // 31)%3]
    alpha = alp_arr[ind % 31]
    
    power = int(sys.argv[2]) 
    fol = sys.argv[3]
    
    torus1 = landau.Torus(Nphi, aspect_ratio = aspect_ratio)
    if power == 1:
        vParams = {'n': 1, 'x': np.inf} # Coulomb
    else:
        vParams = {'n': 2, 'x': 8}
        
    hamParams= {'alpha': alpha, 'n': 0, 'Nphi': Nphi} # mass anisotropy, Landau level index
    
    
    pot1 = landau.Potential()
    pot1.setV2(torus1, vParams, hamParams)
    T4 = np.real(utils.Utils.make4tensorsym(torus1, pot1))
    
    for sector in range(Ne):
    
        hilb0, hilbLen0 = utils.Utils.getHilb(Nphi, Ne, sector)
        hilbert0 = hilbert.Hilbert(Nphi, Ne, sector, hilb0, hilbLen0, T4)
        M0 = LinearOperator((len(hilb0), len(hilb0)), matvec=hilbert0.getMatVec)
        E0, V0 = eigsh(M0, k=6, which='SA')

        fil = 'Nphi{0:d}_Ne{1:d}_sector{2:d}_alpha{3:d}_ar{4:d}'.format(
               Nphi, Ne, sector, int(round(10*alpha)), int(round(100*aspect_ratio)))

        np.save(fol+fil+'_E', E0)
        np.save(fol+fil+'_V', V0)