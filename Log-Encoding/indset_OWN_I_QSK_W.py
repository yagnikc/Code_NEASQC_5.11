def indsetsolver(G,b,mult,num_iter):
    '''
    This function takes as input a Graph and returns the maximum independent set of the graph.
    '''
    import sys
    #sys.path.insert(0,'/data_lab/users/y.chatterjee/environ/lib/python3.6/site-packages/')
    import qiskit
    #help(qiskit)
    import geneticalgorithm as gs
    from qiskit.opflow.primitive_ops import PauliSumOp
    import func_timeout as func_timeout
    import math
    import networkx as nx
    import numpy as np
    from HermitiantoUnitary_OWN_I_QSK_H import decompose

    N=len(G.nodes()) #number of vertices (same as the number of variables)
    nqbits=math.ceil(math.log(2*N,2))
    dimensions=2**nqbits
    sumweights=0
    for x in G.nodes():
        sumweights+=G.nodes.data()[x]['weight']
    p=sumweights/mult
    nodedict={}
    nodecount=0
    for x in G.nodes():
        nodedict[nodecount]=x
        nodecount+=1
    #first term of the qubo
    
    mwis1=np.zeros((2*N,2*N))
    for i in range(N):
        for j in range(N):
            #first term with weights 
            mwis1[i][i+N]= 0.25*G.nodes.data()[nodedict[i]]['weight']#there is a 1/2 in the expression and another 1/2 to make it symmetric
            mwis1[i+N][i]= 0.25*G.nodes.data()[nodedict[i]]['weight']   
            #penalty terms

            #1 quadratic terms
            if (nodedict[i],nodedict[j]) in G.edges() :
                mwis1[i][j]=p/8


    #2 linear terms in the penalty                    
    mwis2=np.zeros((2*N,2*N))
    for i in range(N):
        for j in range(i+1,N):
            if (nodedict[i],nodedict[j]) in G.edges() :
                mwis2[i][i+N]=mwis2[i][i+N]- p/8
                mwis2[j][j+N]=mwis2[j][j+N]- p/8
                mwis2[i+N][i]=mwis2[i+N][i]- p/8
                mwis2[j+N][j]=mwis2[j+N][j]- p/8

    mwis=mwis1+mwis2        
    Hamilt=np.zeros((dimensions,dimensions))
    for i in range(len(mwis)):
        for j in range(len(mwis)):
            Hamilt[i][j]=mwis[i][j]
    b='statevector_simulator'
    #b='ibmq_qasm_simulator'

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
    #for x in Q :
    #    label=x['label']
    #    suma=eval(x['label'][0])
    #    for j in range (1,len(x['label'])):
    #        suma^=eval(x['label'][j])
    #    op=op+float(x['coeff']['real'])*suma
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
    if b =='statevector_simulator' or b=='qasm_simulator' :
        backends = Aer.get_backend(b)
    else :
        backends = provider.get_backend(b)

    def objf(params):
        #print("parameter", params)
        #print("len", len(params))    
        params2=[0 if x<=math.pi else 1 for x in params]  # converting from 0-2pi to 0-1
        qr = QuantumRegister(nqbits,'q'+str(nqbits)) #initialize register in the no ancilla mode
        Q2=QuantumCircuit(qr)
        Q1=QuantumCircuit(qr)   
        GateVec=[]#Initialize a vector of 0 and 1 used for optimization
        for i in range (0,len(params2)):
            GateVec.append(cm.exp((0.0-1.0j)*(params2[i]*np.pi)))
        for i in range(dimensions-len(params2)):
            GateVec.append(cm.exp((0.0-1.0j)*(0*np.pi)))

        for i in range (0,nqbits): #apply Haddamard gate to each qubit
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
        return expval

    #note that this version does not work without variable reduction
    from scipy import optimize
    import math as mp


    class MyBounds(object):
        ''' 
        bounds class to make sure your variable is with in the inspected bounds
        '''
        def __init__(self, xmin, xmax):
            self.xmax = np.array(xmax)
            self.xmin = np.array(xmin)

        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            tmax = bool(np.all(x <= self.xmax))
            tmin = bool(np.all(x >= self.xmin))
            return tmin and tmax

    #print (84 * '-')
    #print ("# Choose the reduction degree ")
    #print("the reduction degree must be between 2 and", str(2**nqbits), "- must be a power of 2")
    #reduction_degree=input('Enter the reduction degree ')
    reduction_degree = 1
    # init bounds
    lower_bounds = []
    upper_bounds = []
    if int(reduction_degree)==1:
        for i in range (0,int(2**nqbits/reduction_degree)-1):
            lower_bounds.append(0.0)
            upper_bounds.append(2*np.pi)
    else: 
        for i in range (0,int(2**nqbits/reduction_degree)):
            lower_bounds.append(0.0)
            upper_bounds.append(2*np.pi)
    my_bounds    = MyBounds(lower_bounds, upper_bounds)

    if mp.log(reduction_degree,2)-int(mp.log(reduction_degree,2))!=0.0:
        print("reduction not a power of 2")
        sys.exit()

    #print("the total number of variational variables is ",2**nqbits/reduction_degree)
    bnd=[] #initial bounds

    if reduction_degree==1:
        for i in range (0,2**nqbits-1): #append bounds 0<=x<=1 for every variable (relax the binary problem)
            bnd.append((0,1.0))
        x0=np.zeros(2**nqbits) #np.zeros(2**nqbits-1) #initial guess is zeros
    else:
        for i in range (0,int(2**(nqbits)/reduction_degree)): #append bounds 0<=x<=1 for every variable (relax the binary problem)
            bnd.append((0,2*np.pi))
            x0=np.zeros(int(2**(nqbits)/reduction_degree)) #np.zeros(2**nqbits-1) #initial guess is zeros
            print("x0",x0)
    def bnd_f(x):       
        return 0 and bnd
    bnd=tuple(bnd)
    bounds=[[0,2*np.pi]]

    #construct the bounds in the form of constraints
    cons = []
    for factor in range(len(bounds)):
        lower, upper = bounds[factor]
        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}
        cons.append(l)
        cons.append(u)
    ops={'maxiter':200} #, 'ftol':1e-5,'eps': 5.0e-1}

 
    #choice = input('Enter your choice of the continual optimizer [1-7] : ')
    choice = 7

    if choice ==1:
        methods='SLSQP'
    if choice ==2:    
        methods='L-BFGS-B'
    if (choice ==3): #Certain versions of the code gave errors for any reduction other than maximal. Ego fails anyway
        print("EGO")
    if choice ==4:
        methods='COBYLA'
    if choice ==5:
        methods='Nelder-Mead'    
    if choice ==6:
        methods='ParticlevSwarm optimization'  
    if choice ==7:
        methods='Genetic algorithm'

    vr=time.time()
    # Only SLSQP and L-BFGS-B methods will work. So choose one and comment out the other
    if (choice ==1 or choice==2):
        ResCon=scipy.optimize.minimize(objf, x0, args=(), method=methods,bounds=bnd,options=ops)

    if choice == 3:
        from smt.applications import EGO
        from smt.sampling_methods import FullFactorial
        auxB=[[0.0, 2*np.pi]]
        xlimits=np.array(auxB)
        for i in range (0,int(2**nqbits/reduction_degree)-1):
            print(i)
            xlimits=np.vstack((xlimits,auxB))       
        print("XL",xlimits) 
        n_doe=15 # number of initial sammpling points
        n_iter =30 # number of iterations after sampling
        count=0
        criterion = "EI"  #'EI' or 'SBO' or 'UCB'

        ResCon = EGO(n_iter=n_iter, criterion=criterion, n_doe=n_doe, xlimits=xlimits)
        x, fun, _, x_data, y_data = ResCon.optimize(fun=objf_ego)
        x=x
        fun=fun[0]
    if choice ==6:
        import pyswarms as ps
        from pyswarms.backend.topology import Pyramid, Star, Ring, VonNeumann, Random
        from pyswarms.utils.functions import single_obj as fx    
        # Set-up hyperparameters and topology
        my_topology = Random(static=False)
        count=0
        npart=5
        niter=15
        options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k':npart-1}    
        # Call instance of GlobalBestPSO
        optimizer = ps.single.GeneralOptimizerPSO(n_particles=npart, dimensions=int(2**nqbits/reduction_degree),
                                        options=options, topology=my_topology)
        # Perform optimization
        fun, x  = optimizer.optimize(objf_pyswarms, iters=niter)
    if choice ==7:
        from geneticalgorithm import geneticalgorithm as ga

        algorithm_param = {'max_num_iteration': num_iter,\
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
    if (choice == 4 or choice==5):
        ResCon=scipy.optimize.minimize(objf, x0, args=(), method=methods, constraints=cons,options=ops)

    execution_time_min=int((time.time()-vr)/60)
    execution_time_sec=(time.time()-vr)%60

    print("the execution duration is", execution_time_min , " minutes, and", execution_time_sec, " sec")

    partition =[0 if 0<i<np.pi else 1 for i in x]  # convertion thetas from pi format to 0 and 1 

    #get the sets divided into

    SS=[]
    SS1=[]
    for i in range(len(partition)):
        if partition[i]==0:
            SS.append(nodedict[i])
        else:
            SS1.append(nodedict[i])  # SS1 now has the node numbers of the Original Graph

    #check independent set        


    f=0
    for x in SS1 :
        for y in SS1 :
            if (x,y) in G.edges():
                #print(x,',',y)
                f=f+1
    print(f/2)
    success=False
    if f>0 :
        success=False
        print('Solution is not an indset')
    else :
        success=True
        print('Solution is an indset')
        
    objfn=0   
    for i in SS1 :
        objfn=objfn+G.nodes.data()[i]['weight']
    print("Objective Function=",objfn)
    print(SS1)
    return objfn, SS1, success

    
    
