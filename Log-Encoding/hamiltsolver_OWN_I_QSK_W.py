

def hamiltsolver(R,b):
    """
    This function takes as input a Hamiltonian Matrix (which is the negative Laplacian Matrix in the case of the MaxCut problem) and returns 
    Maximum Cut of the problem. 

    Input : R : Hamiltonian Matrix
            b : the backend 

    Output : 
        partition : List of 0's and 1's depicting the 2 sets into which the nodes are divided
        SS and SS1 : 2 subsets of nodes of the input graph which form the maximum cut.       
    """
    import sys
    import qiskit
    import geneticalgorithm as gs
    from qiskit.opflow.primitive_ops import PauliSumOp
    import func_timeout as func_timeout
    import math
    import networkx as nx
    import numpy as np
    from HermitiantoUnitary_OWN_I_QSK_H import decompose

    
    N=len(np.diag(R))   #number of the nodes of the graph
    nqbits = math.ceil(math.log(N,2))  #number of qubits is the ceiling of log(N)
    dimensions=2**nqbits   # this is the dimension of the matrix which has to be a power of 2
#    R=nx.linalg.laplacianmatrix.laplacian_matrix(G).toarray()
    Hamilt=np.zeros((dimensions,dimensions))
#    print(Hamilt.size)
    for i in range (0,N):
        for j in range (0,N):
            Hamilt[i,j]=R[i,j]



    #from qiskit import IBMQ
    #IBMQ.save_account('97af5cba035977a5d0a10dab4c47af7c33f305e653dae88fdf3578acf9a8f6447b86bdc027bbc50169f7e4d319f96abd50535d6e645b896236ee79caa37582f5',overwrite=True)
    import os
    os.environ['REQUESTS_CA_BUNDLE'] = os.path.join('/etc/pki/ca-trust/extracted/pem/','tls-ca-bundle.pem') 
    #IBMQ.load_account()
    #provider = IBMQ.get_provider(hub='ibm-q-france', group='univ-montpellier', project='default')
    #provider = IBMQ.get_provider(hub='partner-cqc',group='total-energies',project='combinatorialopt')
    #backends = provider.backends
    #print([backend.name() for backend in IBMQ.providers()[1].backends()])

    import time
    from qiskit.extensions import UnitaryGate
    from tqdm import tqdm
    from qiskit.opflow import X,Y,Z,I,CircuitStateFn
    t0=time.time()
    Q=decompose(Hamilt,nqbits)
    Q=[x for x in Q if isinstance(x, dict)]
    op=0
    #print('start of decomposition')
    paulis=[x['label'] for x in Q]
    weights=[x['coeff'] for x in Q]
    pauli_op = [([pauli,weight]) for pauli,weight in zip(paulis,weights)]
    op = PauliSumOp.from_list([ op for op in pauli_op ])
    t1=time.time()
    #print(op)
    #print('end of decomposition') 
    print("Time to decompose",t1-t0)

    #print("The graph has ", G.number_of_edges(), "edges")
    #print("The matrix has", 4**nqbits, " terms")
    import qiskit
    from qiskit import Aer
    from qiskit.utils import QuantumInstance
    from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn, MatrixExpectation
    from qiskit.circuit.library import RZZGate, rz, x, Diagonal
    from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
    #from qiskit.algorithms import COBYLA
    import scipy.optimize
    import warnings
    #from qiskit.providers import BaseBackend
    #from qiskit.providers.aer.noise import NoiseModel
    import time
    import math as mp
    import numpy as np
    import sympy
    import cmath as cm
    import logging
    #from qiskit.providers.aer.extensions.snapshot_expectation_value import *
    import logging, sys
    from qiskit.providers.ibmq.job import IBMQJob
    from qiskit.compiler import transpile

    logging.disable(sys.maxsize)
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)

    warnings.filterwarnings('ignore') #Don't display warnings - some packages outdated
    if b =='statevector_simulator' or b=='qasm_simulator':
        backends = Aer.get_backend(b)
    #else :
        #backends = provider.get_backend(b)
        
    
    def objf(params):
        #print("parameter", params)
        params2=[0 if x<=math.pi else 1 for x in params] # converting from 0-2pi to 0-1
        #print(params2)
        qr = QuantumRegister(nqbits,'q'+str(nqbits)) #initialize register in the no ancilla mode
        Q2=QuantumCircuit(qr)
        Q1=QuantumCircuit(qr)   
        GateVec=[]#Initialize a vector of 1 and -1 used for optimization
        for i in range (len(params2)):
            GateVec.append(cm.exp((0.0-1.0j)*(params2[i]*np.pi)))
        for j in range(dimensions-len(params2)):
            GateVec.append(1+0j)
        
        for i in range (nqbits): #apply Hadamard gate to each qubit
            Q1.h(i)
        Q2.diagonal(GateVec, qr)
        QMGB=Q1.compose(Q2)
        psi = CircuitStateFn(QMGB) #optimize circuit depth
        q_instance = QuantumInstance(backends, shots=8192)

        # define the state to sample
        measurable_expression = StateFn(op, is_measurement=True).compose(psi) 

        # convert to expectation value
        expectation = MatrixExpectation().convert(measurable_expression)  

        # get state sampler (you can also pass the backend directly)
        sampler = CircuitSampler(q_instance).convert(expectation) 
        expval=0.0#auxilllary counter
        expval=sampler.eval().real
        #print(2**(nqbits-2)*expval)
        return 2**(nqbits-2)*expval
     
    #note that this version does not work without variable reduction
    from scipy import optimize
    import math as mp

    vr=time.time()
    
    from geneticalgorithm import geneticalgorithm as ga

    algorithm_param = {'max_num_iteration': 40,\
                       'population_size':40,\
                       'mutation_probability':0.1,\
                       'elit_ratio': 0.05,\
                       'crossover_probability': 0.5,\
                       'parents_portion': 0.3,\
                       'crossover_type':'uniform',\
                       'max_iteration_without_improv':None}
#0.25

    varbound=np.array([[0,2*np.pi]]*int(N))

    modelGA=ga(function=objf,dimension=int(N),variable_type='real',variable_boundaries=varbound
        ,algorithm_parameters=algorithm_param, function_timeout=100000)

    modelGA.run()
    x=modelGA.best_variable
    fun=modelGA.best_function
    
    execution_time_min=int((time.time()-vr)/60)
    execution_time_sec=(time.time()-vr)%60

    partition =[0 if 0<i<np.pi else 1 for i in x]  # convertion thetas from pi format to 0 and 1 
    spin_partition=[1 if 0<i<np.pi else -1 for i in x]  #1 -1 version of the same partition
    
    #get the 2 sets from the partition

    SS=[]
    SS1=[]
    for i in range(len(partition)):
        if partition[i]==0:
            SS.append(i)
        else:
            SS1.append(i)

    print("The first group of nodes is: ",  SS)
    print("The second group of nodes is: ",  SS1)
    #print("the noise-corrected size of the  cut is: ",nx.cut_size(G,S1))
    
    return partition, SS, SS1    
