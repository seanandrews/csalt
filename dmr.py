import os, sys, time, importlib
import numpy as np
import const as const
from classes import dataset
from csalt_models import vismodel_def as vismodel


# load inputs
inp = importlib.import_module('mconfig_'+sys.argv[-1])

# set parameters
theta = np.array([inp.incl, inp.PA, inp.mstar, inp.r_l, inp.z0, inp.psi, 
                  inp.Tb0, inp.q, inp.Tback, inp.dV0, 
                  inp.vsys, inp.xoff, inp.yoff])
theta_fixed = inp.nu_rest, inp.FOV, inp.Npix, inp.dist, inp.rmax

# load metadata
data_dict = np.load(inp.dataname+'.npy', allow_pickle=True).item()

# calculate
for i in range(data_dict['nobs']):

    # load dataset
    d_ = np.load(inp.dataname+'_EB'+str(i)+'.npz')
    vdata = dataset(d_['um'], d_['vm'], d_['data'], d_['weights'],
                    d_['nu_TOPO'], d_['nu_LSRK'], d_['tstamp_ID'])

    # calculate visibilities
    modelvis = vismodel(theta, theta_fixed, vdata)

    # pack dataset and model back into file
    os.system('rm -rf '+inp.dataname+'_EB'+str(i)+'.npz')
    np.savez_compressed(inp.dataname+'_EB'+str(i)+'.MOD', 
                        model=modelvis, resid=vdata.data - modelvis)


# convert the model and residual visibilities into MS format
os.system('rm -rf CASA_logs/proc_DMR_'+sys.argv[-1]+'.log')
os.system('casa --nologger --logfile CASA_logs/proc_DMR_'+sys.argv[-1]+\
          '.log -c CASA_scripts/proc_DMR.py '+sys.argv[-1])
