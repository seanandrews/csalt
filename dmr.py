import os, sys, time
import numpy as np
import const as const
from csalt_vismodel import *


class vdata:
   def __init__(self, u, v, vis, wgt, nu_topo, nu_lsrk):
        self.u = u
        self.v = v
        self.vis = vis
        self.wgt = wgt
        self.nu_TOPO = nu_topo
        self.nu_LSRK = nu_lsrk


# set target name
targ_name = 'Sz129'



# load inputs
os.system('cp fconfig_'+targ_name+'.py fconfig.py')
import fconfig as inp

# set parameters
theta = np.array([inp.incl, inp.PA, inp.mstar, inp.r_l, inp.z0, inp.psi, 
                  inp.Tb0, inp.q, inp.Tback, inp.dV0, 
                  inp.vsys, inp.xoff, inp.yoff])
theta_fixed = inp.nu_rest, inp.FOV, inp.Npix, inp.dist, inp.r0

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
clog = 'CASA_logs/proc_DMR_'+targ_name+'.log'
os.system('casa --nologger --logfile '+clog+' -c CASA_scripts/proc_DMR.py')

# image the concatenated data products


