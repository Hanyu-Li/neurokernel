#!/usr/bin/env python

"""
Create generic LPU and simple pulse input signal.
"""

from itertools import product
import sys

import numpy as np
import scipy.io as sio
import h5py
import networkx as nx

def create_lpu(file_name, N_sensory, N_local, N_output):
    """
    Create a generic LPU.

    Creates a GEXF file containing the neuron and synapse parameters for an LPU
    containing the specified number of local and projection neurons. The GEXF
    file also contains the parameters for a set of sensory neurons that accept
    external input. All neurons are either spiking or graded potential neurons;
    the Leaky Integrate-and-Fire model is used for the former, while the
    Morris-Lecar model is used for the latter (i.e., the neuron's membrane
    potential is deemed to be its output rather than the time when it emits an
    action potential). Synapses use either the alpha function model or a
    conductance-based model. 

    Parameters
    ----------
    file_name : str
        Output GEXF file name.
    N_sensory : int
        Number of sensory neurons.
    N_local : int
        Number of local neurons.
    N_output : int
        Number of project neurons.
    """
    
    # Set numbers of neurons:
    neu_type = ('sensory', 'local', 'output')
    neu_num = (N_sensory, N_local, N_output)

    # Neuron ids are between 0 and the total number of neurons:
    G = nx.DiGraph()
    G.add_nodes_from(range(sum(neu_num)))

    idx = 0
    spk_out_id = 0
    gpot_out_id = 0
    for (t, n) in zip(neu_type, neu_num):
        for i in range(n):
            name = t+"_"+str(i)
            
            # All local neurons are graded potential only:
            if t != 'local' or np.random.rand() < 0.5:
		V_data = np.random.uniform(-10,-1)

                G.node[idx] = {
                    'model': 'Olfactory_receptor',
                    'name': name+'_s',
                    'extern': True if t == 'sensory' else False,
                    'public': True if t == 'output' else False,
                    'spiking': False,
                    'V': V_data,                    
		    'V_prev': V_data,
                    'X_1': 0.0,
                    'X_2': 0.0,
                    'X_3': 0.0}
		"""
                if t == 'output':
                    G.node[idx]['selector'] = '/gen/out/spk/' + str(spk_out_id)
                    spk_out_id += 1 
		"""
                if t == 'output' or 'sensory':
                    G.node[idx]['selector'] = '/gen/out/gpot/' + str(gpot_out_id)
                    gpot_out_id = 1
            else:
                G.node[idx] = {
                    'model': "MorrisLecar",
                    'name': name+'_g',
                    'extern': True if t == 'sensory' else False,
                    'public': True if t == 'output' else False,
                    'spiking': False,
                    'V1': 0.03,
                    'V2': 0.015,
                    'V3': 0,
                    'V4': 0.03,
                    'phi': 0.025,
                    'offset': 0,
                    'initV': -0.05214,
                    'initn': 0.02, 
                }
                if t == 'output':
                    G.node[idx]['selector'] = '/gen/out/gpot/' + str(gpot_out_id)
                    gpot_out_id = 1
            idx += 1

    # Assume a probability of synapse existence for each group of synapses:
    # sensory -> local, sensory -> output, local -> output, output -> local:            
    for r, (i, j) in zip((0.5, 0.1, 0.1, 0.3),
                         ((0, 1), (0, 2), (1, 2), (2,1))):
        src_off = sum(neu_num[0:i])
        tar_off = sum(neu_num[0:j])
        for src, tar in product(range(src_off, src_off+neu_num[i]),
                                range(tar_off, tar_off+neu_num[j])):

            # Don't connect all neurons:
            if np.random.rand() > r: continue

            # Connections from the sensory neurons use the alpha function model;
            # all other connections use the power_gpot_gpot model:
            name = G.node[src]['name'] + '-' + G.node[tar]['name']
            if G.node[src]['spiking'] is True:
                G.add_edge(src,tar,type='directed',attr_dict={
                    'model'       : 'AlphaSynapse',
                    'name'        : name,
                    'class'       : 0 if G.node[tar]['spiking'] is True else 1,
                    'ar'          : 1.1*1e2,
                    'ad'          : 1.9*1e3,
                    'reverse'     : 65*1e-3 if G.node[tar]['spiking'] else 0.01,
                    'gmax'        : 3*1e-3 if G.node[tar]['spiking'] else 3.1e-4,
                    'conductance' : True})
            else:
                G.add_edge(src,tar,type='directed',attr_dict={
                    'model'       : 'power_gpot_gpot',
                    'name'        : name,
                    'class'       : 2 if G.node[tar]['spiking'] is True else 3,
                    'slope'       : 0.8,
                    'threshold'   : -0.05,
                    'power'       : 1,
                    'saturation'  : 0.03,
                    'delay'       : 1,
                    'reverse'     : -0.1,
                    'conductance' : True})

    nx.write_gexf(G, file_name)

def create_input(file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    N_sensory : int
        Number of sensory neurons.
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """

    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    I  = np.zeros((Nt, N_sensory), dtype=np.float64)
    I[np.logical_and(t>start, t<stop)] = I_max

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)

def create_input_from_mat(file_name, mat_file_name, N_sensory, dt=1e-4, dur=1.0, start=0.3, stop=0.6, I_max=0.6):
    """
    Create input stimulus for sensory neurons in artificial LPU.

    Creates an HDF5 file containing input signals for the specified number of
    neurons. The signals consist of a rectangular pulse of specified duration
    and magnitude.

    Parameters
    ----------
    file_name : str
        Name of output HDF5 file.
    N_sensory : int
        Number of sensory neurons.
    dt : float
        Time resolution of generated signal.
    dur : float
        Duration of generated signal.
    start : float
        Start time of signal pulse.
    stop : float
        Stop time of signal pulse.
    I_max : float
        Pulse magnitude.
    """
    mat = sio.loadmat(mat_file_name)

    Ostim_data = 100*np.array(mat.get('c'), np.float32)
    dt = mat.get('dt')/1000
    dur = dt * Ostim_data.shape[0]





    Nt = int(dur/dt)
    t  = np.arange(0, dt*Nt, dt)

    #I  = np.zeros((Nt, N_sensory), dtype=np.float64)
    I = np.asarray(Ostim_data, np.float32)
    I = I.repeat(10)

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='olfactory_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('in_file_name', nargs='?', default='olfactory_input.h5',
                        help='Input file name')
    parser.add_argument('mat_file_name', nargs='?', default='./2011_03_04_0379.mat',
                        help='Input file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    dt = 1e-4
    dur = 25.0
    start = 0.0
    stop = 1.0
    I_max = 0.6
    #neu_num = [np.random.randint(31, 40) for i in xrange(3)]
    neu_num = [10, 0, 0]

    create_input_from_mat(args.in_file_name, args.mat_file_name, neu_num[0], dt, dur, start, stop, I_max)
    create_lpu(args.lpu_file_name, *neu_num)
