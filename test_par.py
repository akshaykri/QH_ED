import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import numba
import landau
import utils
import hilbert

##################
##################
print(scipy.version.version)

Nphi = 27
Ne = 9
torus1 = landau.Torus(Nphi, aspect_ratio=1.0)
vParams = {'n': 1, 'x': np.inf} # Coulomb
hamParams= {'alpha': 1.0, 'n': 0, 'Nphi': Nphi} # mass anisotropy, Landau level index

t1 = time.time()
pot1 = landau.Potential()
pot1.setV2(torus1, vParams, hamParams)
T4 = np.real(utils.Utils.make4tensorsym(torus1, pot1))

t2 = time.time()
sector = 0
hilb, hilbLen = utils.Utils.getHilb(Nphi, Ne, sector)
dictx = hilbert.getDict(hilb)
dij = hilbert.getMatVec(v, Nphi, Ne, NH, hilb, hilbLen, T4, dictx)
HMat = scipy.sparse.coo_matrix((dij[0, :], 
                      (dij[1, :], dij[2, :])),
                      shape=(sum(hilbLen), sum(hilbLen)))


t3 = time.time()
vIn = np.ones(np.sum(hilbLen))

t4 = time.time()
vOut = HMat.dot(vIn)

t5 = time.time()
E_all, V_all = scipy.sparse.linalg.eigsh(HMat, k=6, which='SA')
t6 = time.time()




print(t2-t1)
print(t3-t2)
print(t4-t3)
print(t5-t4)
print(t6-t5)

##################
##################


# """
# Nphi   : int, number of orbitals
# Ne     : int, number of electrons
# sector : int, momentum sector
# hilb   : np.array, NH x Ne of dtype int8, has the entire Hilbert space 
# hilbLen: size of Hilbert space in each subsector
# T4     : tensor of interaction elements V_{n1 n2 n3 n4} = V_{n1 0 n3 n1-n3}
# """
        
# Nphi = 18
# Ne = 6
# sector = 3
# hilb, hilbLen = utils.Utils.getHilb(Nphi, Ne, sector)
# NH = len(hilb) # size of Hilbert space


# torus1 = landau.Torus(Nphi, aspect_ratio=1.0)
# vParams = {'n': 1, 'x': np.inf} # Coulomb
# hamParams= {'alpha': 1.0, 'n': 0, 'Nphi': Nphi} # mass anisotropy, Landau level index
# pot1 = landau.Potential()
# pot1.setV2(torus1, vParams, hamParams)
# T4 = np.real(utils.Utils.make4tensorsym(torus1, pot1))

# @numba.njit(numba.types.DictType(numba.int64, numba.int64)(numba.int8[:,:]))
# def getDict(hilb):
#     """
#     get a hash table mapping keys (state binary string) to indices
#     """
#     keys = np.sum(2**hilb, axis=1)
#     d = numba.typed.Dict.empty(
#         key_type=numba.int64,
#         value_type=numba.int64,
#         )

#     for i in range(keys.size):
#         d[keys[i]] = i
    
#     return d

# dictx = getDict(hilb)

# def readDictx(n):
#     return dictx[n]

# @numba.njit(numba.float64[:](numba.float64[:]))
# def getMatVec(v):
#     """
#     Hamiltonian operator
#     """
#     vOut = np.zeros(NH)

#     for cHilb in range(NH):
#         eOcc = hilb[cHilb] # occupied electrons

#         diagel = 0.0

#         #find pairs
#         for p1 in numba.prange(Ne):
#             for p2 in numba.prange(p1+1, Ne):
#                 c1 = eOcc[p1] # orbital of first occupied electron
#                 c2 = eOcc[p2] # orbital of second occupied electron

#                 # diagonal terms
#                 diagel += -T4[(c1-c2)%Nphi, (c1-c2)%Nphi] + \
#                            T4[(c2-c1)%Nphi, 0]

#                 # find new pairs
#                 for cx in range(1, Nphi):
#                     c1new = (c1+cx)%Nphi
#                     c2new = (c2-cx)%Nphi
#                     if c1new >= c2new:
#                         continue

# #                         if c1new in eOcc or c2new in eOcc:
# #                             continue

#                     ####################
# #                         numba does not recognize contains (the 'in' keyword)
#                     searchIn = False
#                     for eOrb in eOcc:
#                         if c1new == eOrb or c2new == eOrb:
#                             searchIn = True
#                             break
#                     if searchIn:
#                         continue
#                     ####################

#                     else:
#                         # at this point, we are sure that c2new > c1new,
#                         # and that they are both not in eOcc

#                         # populate eOccNew
# #                         eOccNew = np.zeros(Ne, dtype='int64') # array of length Ne
#                         eOccNew = np.zeros(Ne, dtype=numba.int8) # array of length Ne
#                         cNeOld = 0
#                         state = 0
#                         for cNe in range(Ne):
#                             while cNeOld == p1 or cNeOld == p2:
#                                 cNeOld += 1

#                             if state == 0:
#                                 if cNeOld < Ne and eOcc[cNeOld] < c1new:
#                                     eOccNew[cNe] = eOcc[cNeOld]
#                                     cNeOld += 1
#                                 else:
#                                     eOccNew[cNe] = c1new
#                                     p1new = cNe
#                                     state = 1

#                             elif state == 1:
#                                 if cNeOld < Ne and eOcc[cNeOld] < c2new:
#                                     eOccNew[cNe] = eOcc[cNeOld]
#                                     cNeOld += 1
#                                 else:
#                                     eOccNew[cNe] = c2new
#                                     p2new = cNe
#                                     state = 2

#                             else:
#                                 eOccNew[cNe] = eOcc[cNeOld]
#                                 cNeOld += 1


#                         # find index of eOccNew in hilb, again binary search
#                         indNew = readDictx(np.sum(2**eOccNew)) # self.indexOf(eOccNew)

#                         # (-1)^n = 1 - 2*(n%2)
#                         #the four terms
#                         matrixel = ((1 - 2*((p1new+p2new+p1+p2 + 1)%2)) * 
#                                      T4[(c2new-c1new)%Nphi, (c2-c1new)%Nphi]) + \
#                                    ((1 - 2*((p1new+p2new+p1+p2)%2)) * 
#                                      T4[(c2new-c1new)%Nphi, (c1-c1new)%Nphi])

#                         # update vOut
#                         vOut[indNew] += matrixel * v[cHilb]
                        
#         vOut[cHilb] += diagel * v[cHilb]
#     return vOut
                        
# vIn = np.ones(np.sum(hilbLen))

# t1 = time.time()
# vOut = getMatVec(vIn)
# t2 = time.time()
# print(t2 - t1)

######################
######################

# @numba.njit(numba.float64[:](numba.float64[:]), parallel=False)
# def screwIt1(v):
#     N = len(v)
#     vOut = np.zeros(N)
    
#     for i in numba.prange(N):
#         for j in numba.prange(N):
#             vOut[j] += (i+j)*v[i]
            
#     return vOut

# @numba.njit(numba.float64[:](numba.float64[:]), parallel=True)#, fastmath=False)
# def screwIt2(v):
#     N = len(v)
#     vOut = np.zeros(N)
    
#     for i in numba.prange(N):
#         for j in numba.prange(N):
#             vOut[j] += (i+j)*v[i]
            
#     return vOut

# @numba.njit(numba.float64[:](numba.float64[:]), parallel=True, fastmath=True)

# def screwIt3(v):
#     N = len(v)
#     vOut = np.zeros(N)
    
#     for i in numba.prange(N):
#         for j in numba.prange(N):
#             vOut[j] += (i+j)*v[i]
            
#     return vOut

# vIn = np.ones(100_000)

# t1 = time.time()
# vOut1 = screwIt1(vIn)
# t2 = time.time()
# vOut1 = screwIt1(vIn)
# t3 = time.time()

# vOut2 = screwIt2(vIn)
# t4 = time.time()
# vOut2 = screwIt2(vIn)
# t5 = time.time()

# vOut3 = screwIt3(vIn)
# t6 = time.time()
# vOut3 = screwIt3(vIn)
# t7 = time.time()

# print(t2 - t1)
# print(t3 - t2)
# print(t4 - t3)
# print(t5 - t4)
# print(t6 - t5)
# print(t7 - t6)