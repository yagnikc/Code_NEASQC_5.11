import sys
#sys.path.insert(0,'/data_lab/users/y.chatterjee/environ/lib/python3.6/site-packages/')
#sys.path.append('/data_lab/users/y.chatterjee/qlm_notebooks/notebooks_1.2.1/notebooks/')
#import qiskit

import numpy as np

# We import the tools to handle general Graphs
import networkx as nx

#import other packages
import random
import numpy 
import csv

n=[62, 91, 90, 36, 54, 84, 40, 75]
#n=[random.randint(1,1000) for i in range(N)]
#n=[87, 29, 37, 53, 29, 19, 76, 50, 89, 97, 86, 28, 53, 48, 65, 1, 75, 66, 11, 56, 84, 78, 42, 88, 4, 20, 26, 27, 33, 94, 48, 75]
print(n)
N=len(n)
G=nx.Graph()
for i in range(N):
    G.add_node(i)
for i in range(N):
    for j in range(i+1,N):
        G.add_edge(i,j, weight=int(n[i])*int(n[j]))

R=nx.linalg.laplacianmatrix.laplacian_matrix(G).toarray()

#rseed=30
#reg_gr=3
#G = nx.random_regular_graph(reg_gr, dimensions, seed=random.seed(rseed))
#G=nx.path_graph(dimensions)

#choose from these backends for provider, for Aer use statevector_simulator or qasm_simulator
#['ibmq_qasm_simulator', 'ibmq_montreal', 'ibmq_toronto', 'ibmq_mumbai', 'ibmq_lima', 'ibmq_belem', 'ibmq_quito', 'ibmq_guadalupe', 'simulator_statevector', 'simulator_mps', 'simulator_extended_stabilizer', 'simulator_stabilizer', 'ibmq_jakarta', 'ibmq_manila', 'ibm_hanoi', 'ibm_lagos', 'ibm_cairo', 'ibm_auckland', 'ibm_perth', 'ibm_washington']

from hamiltsolver_OWN_I_QSK_W import hamiltsolver
backends = 'statevector_simulator'
print("Chosen backend is ",backends)
cutlist, set_1, set_2 = hamiltsolver(-R,backends)
w1=0
w2=0
for i in range (0,N):
    if cutlist[i]>0.0:
            w1=w1+int(n[i])
    else:        
            w2=w2+int(n[i])            
print(w1)
print(w2)
print("Difference : ", abs(w2-w1))
#a=[w1,w2,abs(w2-w1)]          
