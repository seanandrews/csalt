import os, sys, time, importlib
import numpy as np


file_prefix = 'simp3-nmm'


### Prepare the data for modeling

# Load the configuration file
inp = importlib.import_module('mconfig_'+file_prefix)

# Load the metadata dictionary
data_dict = np.load(inp.dataname+'.npy', allow_pickle=True).item()
nobs = data_dict['nobs']

# Initiate the prepared dictionary
output_dict = {'nobs': nobs, 'chbin': inp.chbin}

# loop through each EB
for i in range(nobs):

    # load the dataset
    d_ = np.load(inp.dataname+'_EB'+str(i)+'.npz')
    idata = dataset(d_['um'], d_['vm'], d_['data'], d_['weights'],
                    d_['nu_TOPO'], d_['nu_LSRK'], d_['tstamp_ID'])

    # convert the LSRK frequency grid to a velocity grid
    vLSRK_grid = const.c_ * (1 - inu_lsrk / restfreq)

    # swap inputs vra if velocities decrease with channel index
    if np.any(np.diff(inu_topo) > 0):
        ivis, iwgt = ivis[:,::-1,:], iwgt[:,::-1,:]
        vLSRK_grid = vLSRK_grid[:,::-1]

    # compute the representative velocities (at EB midpoint)
    mid_stamp = vLSRK_grid.shape[0] / 2
    if (mid_stamp % 1) != 0:
        ilo = np.int(mid_stamp + mid_stamp % 1)
        ivel = np.mean(vLSRK_grid[ilo-1:ilo+1,:], axis=0)
    else:
        ivel = vLSRK_grid[np.int(mid_stamp),:]


