#!/usr/bin/env python
# coding: utf-8

# In[3]:


import sys
import numpy as np
from time import sleep
from tqdm import tqdm
def HS(M1, M2):
    """Hilbert-Schmidt-Product of two matrices M1, M2"""
    return (np.dot(M1.conjugate().transpose(), M2)).trace()



def c2s(c):
    """Return a string representation of a complex number c"""
    if c == 0.0:
        return "0"
    if c.imag == 0:
        return "%g" % c.real
    elif c.real == 0:
        return "%gj" % c.imag
    else:
        return "%g+%gj" % (c.real, c.imag)




# In[ ]:


def decompose(H,nqbits):
    """auxialary counter with 4**nqbits colums of length nqbits with all combination of integers from 0 to 3
    for instance 0,1,2,2; 0,0,0,0; 3,2,1,0; 3301; etc
    This is used instead of using nqbit for loops in the original code"""
    indi=np.zeros((4**nqbits,nqbits))

    for i in tqdm(range (0,4**nqbits)):
        for j in range (0,nqbits):
            if j!=0:
                indi[i,j]=int(i/4**(nqbits-j-1))%4
            else:
                indi[i,j]=int(i/4**(nqbits-1))
    """Decompose Hermitian H into Pauli matrices"""
    """define the Pauli matrices as arrays """
    from numpy import kron
    sx = np.array([[0, 1], [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    # sp = np.array([[0, 0], [1, 0]], dtype=np.complex128)
    # sm = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    id = np.array([[1, 0], [ 0, 1]], dtype=np.complex128)
    S = [id, sx, sy, sz]
    labels = ['I', 'X', 'Y', 'Z']
    # S = [id, sp, sm, sz]
    # labels = ['I', 'P', 'M', 'Z']
    arrs=[]
    for i in tqdm(range (0,4**nqbits)):
        label=""
        for j in range (0,nqbits):
            label=label+labels[int(indi[i,j])]
        aux_mat=[]
        j=nqbits-1
        while j>=1:
            if j==nqbits-1:
                aux_mat=kron(S[int(indi[i,j-1])],S[int(indi[i,j])])
            else:
                aux_mat=kron(S[int(indi[i,j-1])],aux_mat)
            j-=1
        a=HS(H,aux_mat)/(2**nqbits)
        if a != 0:
            #arrs.append({'label': label,'coeff': {'real': float(complex(c2s(a)).real), 'imag': float(complex(c2s(a)).imag)}})
            arrs.append({'label': label,'coeff': float(complex(c2s(a)).real)})
    return arrs #gives a dictionary, transfering the Hermitian graph Laplacian to sum of unitaries
    #which a QC can understand

