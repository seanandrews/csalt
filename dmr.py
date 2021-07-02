import os, sys, time, importlib
import numpy as np
import const as const
from csalt_vismodel import *


# a data class for easier packaging
class vdata:
   def __init__(self, u, v, vis, wgt, nu_topo, nu_lsrk):
        self.u = u
        self.v = v
        self.vis = vis
        self.wgt = wgt
        self.nu_TOPO = nu_topo
        self.nu_LSRK = nu_lsrk


# load inputs
inp = importlib.import_module('fconfig_'+sys.argv[-1])

# set parameters
theta = np.array([inp.incl, inp.PA, inp.mstar, inp.r_l, inp.z0, inp.psi, 
                  inp.Tb0, inp.q, inp.Tback, inp.dV0, 
                  inp.vsys, inp.xoff, inp.yoff])
theta_fixed = inp.nu_rest, inp.FOV, inp.Npix, inp.dist, inp.rmax

# load metadata
data_dict = np.load(inp.dataname+'.npy', allow_pickle=True).item()

# calculate
for i in range(data_dict['nobs']):
    print('EB'+str(i))

    # load dataset
    d_ = np.load(inp.dataname+'_EB'+str(i)+'.npz')
    dataset = vdata(d_['u'], d_['v'], d_['data'], d_['weights'],
                    d_['nu_TOPO'], d_['nu_LSRK'])

    # calculate visibilities
    modelvis = csalt_vismodel(dataset, theta, theta_fixed)

    # pack dataset and model back into file
    os.system('rm -rf '+inp.dataname+'_EB'+str(i)+'.npz')
    np.savez_compressed(inp.dataname+'_EB'+str(i), u=d_['u'], v=d_['v'], 
                        data=d_['data'], weights=d_['weights'], 
                        nu_TOPO=d_['nu_TOPO'], nu_LSRK=d_['nu_LSRK'],
                        model=modelvis)


# convert the model and residual visibilities into MS format
os.system('rm -rf CASA_logs/proc_DMR_'+sys.argv[-1]+'.log')
os.system('casa --nologger --logfile CASA_logs/proc_DMR_'+sys.argv[-1]+\
          '.log -c CASA_scripts/proc_DMR.py '+sys.argv[-1])
