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

def create_lpu(file_name, receptor_type, N_port_in_gpot, fan_out, verbose=False):
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
    neu_type = ('port_in_gpot','sensory')
    neu_num = (N_port_in_gpot, N_port_in_gpot*fan_out)

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
            #if t != 'local' or np.random.rand() < 0.5:
            if t == 'port_in_gpot':
                G.node[idx] = {
                    'model': 'port_in_gpot',
                    'name': name+'_s',
                    'extern': False,
                    'selector': '/%s/%d' % (receptor_type[i],i)}
            elif t == 'sensory':
                V_data = np.random.uniform(-10,-1)
                G.node[idx] = {
                    'model': 'Olfactory_receptor',
                    'name': name+'_s',
                    'extern': True if t == 'sensory' else False,
                    'public': True if t == 'output' else False,
                    'spiking': True,
                    'V': V_data,
                    'V_prev': V_data,
                    'X_1': 0.0,
                    'X_2': 0.0,
                    'X_3': 0.0}

		if t == 'output' or 'sensory':
		    G.node[idx]['selector'] = '/olfactory_receptor/out/spk/' + str(spk_out_id)
		    spk_out_id += 1
            idx += 1
        if verbose: print idx

    # Assume a probability of synapse existence for each group of synapses:
    # sensory -> local, sensory -> output, local -> output, output -> local:
    """
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

    """
    src_off = sum(neu_num[0:0])
    tar_off = sum(neu_num[0:1])
    fan_out = neu_num[1]/neu_num[0]
    for src in range(src_off, src_off+neu_num[0]):
        for tar in range(tar_off+src*fan_out,tar_off+(src+1)*fan_out):
            #print src, tar
            name = G.node[src]['name'] + '-' + G.node[tar]['name']
            if G.node[src]['model'] == 'port_in_gpot':
                G.add_edge(src,tar,type='directed',attr_dict={
                    'model'       : 'dummy_synapse',
                    'name'        : name,
                    'class'       : 3,
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

def create_input_from_mat(file_name, mat_file_name, N_sensory, dt=1e-4, dur=1.0):
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
    I = I.repeat(N_sensory)

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('array', (Nt, N_sensory),
                         dtype=np.float64,
                         data=I)
def parse_receptor_type(receptor_type_file_name):
    f = open(receptor_type_file_name)
    receptor_type = f.read().splitlines()
    return receptor_type
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lpu_file_name', nargs='?', default='olfactory_lpu.gexf.gz',
                        help='LPU file name')
    parser.add_argument('in_file_name', nargs='?', default='olfactory_input.h5',
                        help='Input file name')
    parser.add_argument('mat_file_name', nargs='?', default='./m/2011_03_09_0005.mat',
                        help='Mat file name')
    parser.add_argument('receptor_type_file_name', nargs='?', default='./receptor_type.txt',
                        help='Receptor Type file name')
    parser.add_argument('-s', type=int,
                        help='Seed random number generator')
    args = parser.parse_args()

    if args.s is not None:
        np.random.seed(args.s)
    dt = 1e-4
    dur = 25.0
    #neu_num = [np.random.randint(31, 40) for i in xrange(3)]
    fan_out = 5
    receptor_type = parse_receptor_type(args.receptor_type_file_name)
    port_in_gpot_num = len(receptor_type)
    neu_num = port_in_gpot_num*fan_out


    create_input_from_mat(args.in_file_name, args.mat_file_name,neu_num, dt, dur)
    create_lpu(args.lpu_file_name, receptor_type, port_in_gpot_num, fan_out)
