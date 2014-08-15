from baseneuron import BaseNeuron

import numpy as np
import pycuda.gpuarray as garray
from pycuda.tools import dtype_to_ctype
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
#from neurokernel.LPU.utils.curand import curand_setup

import tables
from jinja2 import Template

class Olfactory_receptor_spk(BaseNeuron):
    def __init__(self, n_dict, spk, dt , debug=False, LPU_id=None):
        self.num_neurons = len(n_dict['id'])
        self.LPU_id = None
        super(Olfactory_receptor_spk, self).__init__(n_dict, spk, dt, debug, LPU_id)
        self.debug = debug

        self.dt = dt
        #self.steps = max(int(round(dt / 1e-5)),1)
        #self.ddt = dt / self.steps

	self.I_drive = garray.to_gpu(np.asarray(np.zeros((self.num_neurons,1)), dtype=np.double))

	binding_rate_data = np.random.rand(self.num_neurons, 1);
	self.binding_rate = garray.to_gpu(np.asarray(binding_rate_data, dtype=np.double))
	init_state = np.zeros((self.num_neurons,13))
	init_state[:,4] = 1e-5
	init_state[:,7] = 1
	self.state = garray.to_gpu(np.asarray(init_state, dtype=np.double))


        #self.V = V
	self.spk = spk
	self.spk_flag = 0
	self.V       = garray.to_gpu(np.asarray(n_dict['V'],       dtype=np.double))
	#self.spk = garray.to_gpu(np.asarray(np.zeros((self.num_neurons,1)), dtype=np.int32))
	self.V_prev  = garray.to_gpu(np.asarray(n_dict['V_prev'],  dtype=np.double))
	self.X_1     = garray.to_gpu(np.asarray(n_dict['X_1'],     dtype=np.double))
	self.X_2     = garray.to_gpu(np.asarray(n_dict['X_2'],     dtype=np.double))
	self.X_3     = garray.to_gpu(np.asarray(n_dict['X_3'],     dtype=np.double))
	#self.I_ext = garray.to_gpu(np.asarray([1]*self.num_neurons, dtype=np.double))
	
        #cuda.memcpy_htod(int(self.V), np.asarray(n_dict['V'], dtype=np.double))
        #cuda.memcpy_htod(int(self.spk), np.asarray(np.zeros((self.num_neurons,1)), dtype=np.double))
	self.update_olfactory_transduction = self.get_olfactory_transduction_kernel()
	self.update_hhn = self.get_multi_step_hhn_kernel()

        if self.debug:
            if self.LPU_id is None:
                self.LPU_id = "anon"
            self.I_file = tables.openFile(self.LPU_id + "_I.h5", mode="w")
            self.I_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))
	    
            self.V_file = tables.openFile(self.LPU_id + "_V.h5", mode="w")
            self.V_file.createEArray("/","array", \
                                     tables.Float64Atom(), (0,self.num_neurons))


    @property
    def neuron_class(self): return True

    def eval(self, st=None):
	self.update_olfactory_transduction.prepared_async_call(self.grid, self.block,st, self.dt, self.I.gpudata, self.binding_rate.gpudata, self.state.gpudata, self.I_drive.gpudata)
	 
	self.update_hhn.prepared_async_call(self.grid, self.block, st, self.spk, self.num_neurons, self.dt*1000,self.I_drive.gpudata,self.X_1.gpudata, self.X_2.gpudata, self.X_3.gpudata, self.V.gpudata, self.V_prev.gpudata)
	"""
	self.spk_flag = 0
	for i in range(10):
	    self.update_hhn.prepared_async_call(self.grid, self.block, st, self.spk, self.num_neurons, self.dt*100,self.I_drive.gpudata,self.X_1.gpudata, self.X_2.gpudata, self.X_3.gpudata, self.V.gpudata, self.V_prev.gpudata)
	    if self.spk == 1:
		self.spk_flag = 1
	self.spk = self.spk_flag
	"""
		

        if self.debug:
            self.I_file.root.array.append(self.I.get().reshape((1,-1)))
            self.V_file.root.array.append(self.V.get().reshape((1,-1)))


    """
    @property
    def update_I_override(self): return True

    def update_I(self, synapse_state, st=None):
        self.I.fill(0)
        if self._pre.size>0:
            self._update_I_non_cond.prepared_async_call(self._grid_get_input,\
                self._block_get_input, st, int(synapse_state), \
                self._cum_num_dendrite.gpudata, self._num_dendrite.gpudata, self._pre.gpudata,
                self.I.gpudata)
        if self._cond_pre.size>0:
            self._update_I_cond.prepared_async_call(self._grid_get_input,\
                self._block_get_input, st, int(synapse_state), \
                self._cum_num_dendrite_cond.gpudata, self._num_dendrite_cond.gpudata,
                self._cond_pre.gpudata, self.I.gpudata, self.V.gpudata, \
                self._V_rev.gpudata)
        



    """
    def get_olfactory_transduction_kernel(self):
	
	template = Template("""
    #include "stdio.h"
    #define NUM_OF_NEURON {{num_neurons}}
    #define cap 4.299e-3          // capacitance, [4.299e-3nF]
    #define cc1lin 1.224          // Ca2+ association rate with CaM, [s^-1]
    #define cc2 22.89             // CaCaM dissociation rate into Ca and CaM, [s^-1]
    #define ck1lin 12.72          // CaMK activation rate by CaCaM, [s^-1]
    #define ck2 0.5564            // CaMK deactivation rate, [s^-1]
    #define clmax 1.013           // maximal g of Cl(Ca) channels, [1.013nS]
    #define cnmax 1.277           // maximal g of CNG channels, [1.277nS]
    #define cx1lin 1.171          // IX activation rate by Ca2+, [s^-1]
    #define cx2 16.12             // IX deactivation rate, [s^-1]
    #define ef 2.162              // Ca2+ extrusion rate constant by NCX, [s^-1]
    #define F 9.649e4             // Faraday's constant, [C/mol]
    #define Cvol 4.2191e-7        // ciliary volume, []
			     // Calculated from Fvol=4.071e-2, table 5
    #define gl 4.575              // leak current conductance, [4.575nS]
    #define Gtot 1                // total number of G-proteins, [part]
    #define hmc1 1.23             // cAMP concentration needed to achieve half-maximal
			     // activation of the CNG channel, [1.23uM]
    #define hmc2 2.604            // Ca2+ concentration needed to achieve half-maximal
			     // activation of the Cl(Ca) channel, [2.604uM]
    #define inf 1.26              // CNG current carried by Ca, [1.26uM*pC^-1]
    #define inhmax 1.396          // maximal CNG channel inhibition factor
    #define k1 0.02351            // odorant binding rate to receptor, [0.02351(uM*s)^-1]
    #define k2 9.915              // G-protein activation rate per bound receptor complex, [s^-1]
    #define kI 0.7037             // IX concentration needed to exert a half-maximal
			     // inhibitory effect (IC_50), [0.7037uM]
    #define kinh 0.3901           // aCaMK concentration needed for half-maximal 
			     // inhibition (IC_50) of cAMP production, [0.3901uM]
    #define kinhcng 0.8951        // CaCaM concentration needed for half-maximal
			     // inhibition of the CNG channel, [0.8951uM]
    #define n1 1.639              // Hill coeff. of the CNG ch. activation function
    #define n2 2.276              // Hill coeff. of the Cl(Ca) ch. activation function
    #define nI 3.705              // Steepness of the decreasing sigmoid representing
			     // IX-mediated inhibition
    #define ninh 1.372            // Steepness of the decreasing sigmoid representing
			     // aCaMK-mediated inhibition of cAMP synthesis
    #define ninhcng 1.112         // Steepness of the sigmoid inhcng representing
			     // the fold increase in K_1/2 of the CNG channel
			     // as a function of CaCaM concentration
    #define pd 10.88              // cAMP molecule degradation rate, [s^-1]
    #define r1 6.911              // odorant unbinding rate from receptor, [s^-1]
    #define r2 4.055              // G-protein deactivation rate, [s^-1]
    #define Rtot 1                // total number of receptor, [part]
    #define smax 91.09            // maximal cAMP production rate by adenylyl cyclase 
			     // per aG, [91.09uM/s]
    #define vcl -11.16            // ClCa channel reversal potential, [11.16mV]
    #define vcng 0.002655         // CNG channel reversal potential, [0.002655mV]
    #define vl -69.67             // effective leak reversal potential, [69.67mV]

    // state variables: bLR,aG,cAMP,Ca,CaCaM,aCaMK,IX,inhcng,I_cng,I_cl,I_ncx,I_leak, V
    __global__ void dougherty_transduction({{type}} dt, {{type}}* Ostim, {{type}} *binding_rate, {{type}} (*state)[13], {{type}} *I)
    {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if(tid < NUM_OF_NEURON){
	    float a1 = 0;
	    float a2 = 0;
	    float a3 = 0;
	    float a4 = 0;
	    float a5 = 0;
	    float a6 = 0;
	    float a7 = 0;
	
	    float b1 = 0;
	    float b2 = 0;
	    float b3 = 0;
	    float b4 = 0;
	    float b5 = 0;
	    float b6 = 0;
	    float b7 = 0;
	
	    float D1 = 0;
	    float D2 = 0;
	    float D3 = 0;
	    float D4 = 0;
	    float D5 = 0;
	    float D6 = 0;
	    float D7 = 0;
	
	    float bLR = state[tid][0];
	    float aG = state[tid][1];
	    float cAMP = state[tid][2];
	    float Ca = state[tid][3];
	    float CaCaM = state[tid][4];
	    float aCaMK = state[tid][5];
	    float IX = state[tid][6];
	    float inhcng = state[tid][7];
	    float I_cng = state[tid][8];
	    float I_cl = state[tid][9];
	    float I_ncx = state[tid][10];
	    float I_leak = state[tid][11];
	    //float I = 0;
	    float V = state[tid][12];
	
	    
	    
	    //numeric method
	    
	    //CaCaM = 1e-5;
	    //inhcng = 1;
	
	    a1 = k1*binding_rate[tid]*Ostim[tid] * Rtot;
	    a2 = k2*bLR * Gtot;
	    a3 = aG*smax / (1 + powf((aCaMK/kinh),ninh));
	    a4 = inf*I_cng + cc2*CaCaM;
	    a5 = cc1lin*Ca;
	    a6 = ck1lin*CaCaM;   
	    a7 = cx1lin * Ca;
	       
	    b1 = k1*binding_rate[tid]*Ostim[tid] + r1;
	    b2 = k2*bLR + r2;    
	    b3 = pd;
	    b4 = cc1lin + ef/(1 + powf((IX/kI),nI));
	    b5 = cc2;
	    b6 = ck2;
	    b7 = cx2; 
	    
	    D1 = exp(-b1*dt);
	    D2 = exp(-b2*dt);
	    D3 = exp(-b3*dt);
	    D4 = exp(-b4*dt);      
	    D5 = exp(-b5*dt);
	    D6 = exp(-b6*dt);
	    D7 = exp(-b7*dt);
	    
	       
	    // feedback model
	    state[tid][0] = bLR*D1 + (a1/b1)*(1-D1);
	    state[tid][1] = aG*D2 + (a2/b2)*(1-D2);
	    
	    
	    // Kurahashi and Menini Experiment
	    // Don't forget to comment out a3, b3, D3, cAMP
	    //     if ( i< 40001 || i>50001 )
	    //         a3 = aG[:,i-1]*smax / (1 + (aCaMK[:,i-1]/kinh)**ninh);
	    //         b3 = pd;
	    //         D3 = np.math.exp(-b3*dt);
	    //         cAMP[:,i] = cAMP[:,i-1]*D3 + (a3/b3)*(1-D3);
	    //     
	    //     else
	    //         cAMP[:,i] = 1.5;
	    //     end
	    
	    
	    
	    state[tid][2] = cAMP*D3 + (a3/b3)*(1-D3);
	    
	    state[tid][3] = Ca*D4 + (a4/b4)*(1-D4);    
	    state[tid][4] = CaCaM*D5 + (a5/b5)*(1-D5);
	    state[tid][5] = aCaMK*D6 + (a6/b6)*(1-D6);
	    state[tid][6] = IX*D7 + (a7/b7)*(1-D7);
	    
	    
	    // calculate cell currents for the feedback model
	    state[tid][8] = cnmax * powf(cAMP,n1) /( powf(cAMP,n1)+powf((inhcng*hmc1),n1) ) * ( vcng - V );    
	    state[tid][9] = clmax * powf(Ca,n2) / (powf(Ca,n2) + powf(hmc2,n2)) * (vcl - V);
	    state[tid][10] = F * Cvol * ef*Ca/( 1+powf((IX/kI),nI) );    
	    state[tid][11] = gl * (vl - V);
	    
	    
	    // get the membrane voltage
	    state[tid][12] = V + 1/cap*dt*( I[tid] + I_leak );       
	    I[tid] = (I_cng + I_cl + I_ncx);
	    // calculate the CNG channel inhibition factor
	    state[tid][7] = 1 + (inhmax-1)*powf(CaCaM,ninhcng) /(powf(CaCaM,ninhcng) + powf(kinhcng,ninhcng));
	    
	    
	
	}
    }
    """)
	dtype = np.double
	scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
	#mod = SourceModule( template2.render(num_of_neuron=self.num_neuron), options=["--ptxas-options=-v"])
	#mod = SourceModule( template, options=["--ptxas-options=-v"])
	self.block = (128,1,1)
	self.grid = ((self.num_neurons - 1) / 128 + 1, 1)
	mod = SourceModule( template.render(type=dtype_to_ctype(dtype),num_neurons=self.num_neurons), options=["--ptxas-options=-v"])

	func = mod.get_function("dougherty_transduction")
	func.prepare([scalartype, #dt
		      np.intp,	  #Ostim/I
		      np.intp,	  #binding_rate
		      np.intp,	  #state
		      np.intp])	  #I_drive
	return func

    # <codecell>

    def get_hhn_kernel(self):
	template = Template("""
	#define g_Na 120.0
	#define g_K  36.0
	#define g_L  0.3
	#define E_K  (-12.0)
	#define E_Na 115.0
	#define E_L  10.613

	__global__ void
	hhn_model(int *spk, int num_neurons, {{type}} dt, {{type}}* I_pre,  \
		  {{type}}* X_1, {{type}}* X_2, {{type}}* X_3, {{type}}* g_V, {{type}}* V_prev)
	{
	    int cart_id = blockIdx.x * blockDim.x + threadIdx.x;

	    if(cart_id < num_neurons)
	    {
		{{type}} V = g_V[cart_id];
		{{type}} bias = 10;
		spk[cart_id] = 0;

		{{type}} a[3];

		a[0] = (10-V)/(100*(exp((10-V)/10)-1));
		X_1[cart_id] = a[0]*dt - X_1[cart_id]*(dt*(a[0] + exp(-V/80)/8) - 1);
	       
		a[1] = (25-V)/(10*(exp((25-V)/10)-1));
		X_2[cart_id] = a[1]*dt - X_2[cart_id]*(dt*(a[1] + 4*exp(-V/18)) - 1);
	       
		a[2] = 0.07*exp(-V/20);
		X_3[cart_id] = a[2]*dt - X_3[cart_id]*(dt*(a[2] + 1/(exp((30-V)/10)+1)) - 1);

		V = V + dt * (I_pre[cart_id]+bias - \
		   (g_K * pow(X_1[cart_id], 4) * (V - E_K) + \
		    g_Na * pow(X_2[cart_id], 3) * X_3[cart_id] * (V - E_Na) + \
		    g_L * (V - E_L)));

		if(V_prev[cart_id] <= g_V[cart_id] && g_V[cart_id] > V) {
		    spk[cart_id] = 1;
		}
		
		V_prev[cart_id] = g_V[cart_id];
		g_V[cart_id] = V;
	    }
	}
	""")
	
	dtype = np.double
	scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
	#hhn_update_block = (128,1,1)
	#hhn_update_grid = ((num_neurons - 1) / 128 + 1, 1)
	self.block = (128,1,1)
	self.grid = ((self.num_neurons - 1) / 128 + 1, 1)
	mod = SourceModule(template.render(type=dtype_to_ctype(dtype)), options=["--ptxas-options=-v"])
	func = mod.get_function("hhn_model")
	
	func.prepare([np.intp,       # spk
		      np.int32,      # num_neurons
		      scalartype,     # dt
		      np.intp,        # I_pre
		      np.intp,        # X1
		      np.intp,        # X2
		      np.intp,        # X3
		      np.intp,        # g_V                 
		      np.intp])       # V_pre

	return func

    def get_multi_step_hhn_kernel(self):
	template = Template("""
	#include <stdio.h>
	#define g_Na 120.0
	#define g_K  36.0
	#define g_L  0.3
	#define E_K  (-12.0)
	#define E_Na 115.0
	#define E_L  10.613

	__global__ void
	hhn_model(int *spk, int num_neurons, {{type}} _dt, {{type}}* I_pre,  \
		  {{type}}* X_1, {{type}}* X_2, {{type}}* X_3, {{type}}* g_V, {{type}}* V_prev)
	{
	    int cart_id = blockIdx.x * blockDim.x + threadIdx.x;

	    if(cart_id < num_neurons)
	    {
		{{type}} V = g_V[cart_id];
		{{type}} bias = 10;
		spk[cart_id] = 0;

		int step = 1;
		{{type}} dt = 0.01;
		if(_dt > dt)
		{
		    step = _dt/dt;
		}

		{{type}} a[3];

		int spk_count = 0;
		//spk[cart_id] = 1;
		//return;

		for(int i=0;i<step;i++){
		    V = g_V[cart_id];
		    a[0] = (10-V)/(100*(exp((10-V)/10)-1));
		    X_1[cart_id] = a[0]*dt - X_1[cart_id]*(dt*(a[0] + exp(-V/80)/8) - 1);
		   
		    a[1] = (25-V)/(10*(exp((25-V)/10)-1));
		    X_2[cart_id] = a[1]*dt - X_2[cart_id]*(dt*(a[1] + 4*exp(-V/18)) - 1);
		   
		    a[2] = 0.07*exp(-V/20);
		    X_3[cart_id] = a[2]*dt - X_3[cart_id]*(dt*(a[2] + 1/(exp((30-V)/10)+1)) - 1);

		    V = V + dt * (I_pre[cart_id]+bias - \
		       (g_K * pow(X_1[cart_id], 4) * (V - E_K) + \
			g_Na * pow(X_2[cart_id], 3) * X_3[cart_id] * (V - E_Na) + \
			g_L * (V - E_L)));

		    if(V_prev[cart_id] <= g_V[cart_id] && g_V[cart_id] > V) {
			spk[cart_id] = 1;
		    }
		    
		    V_prev[cart_id] = g_V[cart_id];
		    g_V[cart_id] = V;
		}
	    }
	}
	""")
	
	dtype = np.double
	scalartype = dtype.type if dtype.__class__ is np.dtype else dtype
	#hhn_update_block = (128,1,1)
	#hhn_update_grid = ((num_neurons - 1) / 128 + 1, 1)
	self.block = (128,1,1)
	self.grid = ((self.num_neurons - 1) / 128 + 1, 1)
	mod = SourceModule(template.render(type=dtype_to_ctype(dtype)), options=["--ptxas-options=-v"])
	func = mod.get_function("hhn_model")
	
	func.prepare([np.intp,       # spk
		      np.int32,      # num_neurons
		      scalartype,     # dt
		      np.intp,        # I_pre
		      np.intp,        # X1
		      np.intp,        # X2
		      np.intp,        # X3
		      np.intp,        # g_V                 
		      np.intp])       # V_pre

	return func
