import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import sys

import landau
import utils
import hilbert

if __name__ == "__main__":
    
    ind = int(sys.argv[1])
    LL = int(sys.argv[2]) 
    fol = sys.argv[3]
    
    alp_arr = np.r_[1.0:4.01:0.1] # 31
    asp_arr = np.array([0, 0.25, 0.5]) # 3
    Ne_arr = np.array([6, 7, 8, 9, 10]) # 5
    # 0 - 464
    
    
    
    Ne = Ne_arr[ind // 93]
    Nphi = 3*Ne
    ar_factor = asp_arr[(ind // 31)%3]
    alpha = alp_arr[ind % 31]
    aspect_ratio = 1/(alpha**(ar_factor))
    
    torus1 = landau.Torus(Nphi, aspect_ratio = aspect_ratio)
    vParams = {'n': 1, 'x': np.inf} # Coulomb
    hamParams= {'alpha': alpha, 'n': LL, 'Nphi': Nphi} # mass anisotropy, Landau level index
    
    
    pot1 = landau.Potential()
    pot1.setV2(torus1, vParams, hamParams)
    T4 = np.real(utils.Utils.make4tensorsym(torus1, pot1))
    
    for sector in range(Ne):
    
        hilb0, hilbLen0 = utils.Utils.getHilb(Nphi, Ne, sector)
        hilbert0 = hilbert.Hilbert(Nphi, Ne, sector, hilb0, hilbLen0, T4)
        M0 = LinearOperator((len(hilb0), len(hilb0)), matvec=hilbert0.getMatVec)
        E0, V0 = eigsh(M0, k=6, which='SA')

        fil = 'Nphi{0:d}_Ne{1:d}_sector{2:d}_alpha{3:d}'.format(
               Nphi, Ne, sector, int(round(10*alpha)))

        np.save(fol+'LL{0:d}/ar{1:03d}/'.format(
                LL, int(round(100*ar_factor)))+fil+'_E', E0)
        np.save(fol+'LL{0:d}/ar{1:03d}/'.format(
                LL, int(round(100*ar_factor)))+fil+'_V', V0)