import numpy as np
import scipy.sparse as ssp
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import LinearOperator
import sys
import time
from functools import partial

import landau
import utils
import hilbert

if __name__ == "__main__":
    
    t1 = time.time()
    ind = int(sys.argv[1])
    LL = int(sys.argv[2]) 
    Ne = int(sys.argv[3])
    fol = sys.argv[4]
    sector = int(sys.argv[5])
    direct = True
    
    alp_arr = np.r_[1.0:4.01:0.1] # 31
    asp_arr = np.array([0, 0.25, 0.5]) # 3
#     Ne_arr = np.array([6, 7, 8, 9, 10]) # 5
    # 0 - 92

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
    t2 = time.time()
    
    print(t2-t1)
    for sector in [sector]:#range(Ne):
    
        hilb0, hilbLen0 = utils.Utils.getHilb(Nphi, Ne, sector)
        dictx = hilbert.getDict(hilb0)
        
        NH = len(hilb0)
        
        if direct:
            dij = hilbert.getMatAux(Nphi, Ne, NH, hilb0, hilbLen0, T4, dictx, 32)
#             HMat = hilbert.dijToCsr(dij, NH)
        
        
        else:
            newMatVec = partial(hilbert.getMatVec, Nphi=Nphi, Ne=Ne, NH=NH, hilb=hilb0, 
                    hilbLen=hilbLen0, T4=T4, dictx=dictx)
            HMat = ssp.linalg.LinearOperator((NH, NH), matvec=newMatVec)
        
        t3 = time.time()
        print(t3-t2)
#         E0, V0 = eigsh(HMat, k=6, which='SA')
        
#         t2 = time.time()
#         fil = 'Nphi{0:d}_Ne{1:d}_sector{2:d}_alpha{3:d}'.format(
#                Nphi, Ne, sector, int(round(10*alpha)))

#         np.save(fol+'LL{0:d}/ar{1:03d}/'.format(
#                 LL, int(round(100*ar_factor)))+fil+'_E', E0)
#         np.save(fol+'LL{0:d}/ar{1:03d}/'.format(
#                 LL, int(round(100*ar_factor)))+fil+'_V', V0)
        print(t2-t3)