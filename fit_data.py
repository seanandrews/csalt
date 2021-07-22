import os, sys, time, importlib
import numpy as np
import const as const


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

    # load the data into a dataset object
    d_ = np.load(inp.dataname+'_EB'+str(i)+'.npz')
    idata = dataset(d_['um'], d_['vm'], d_['data'], d_['weights'],
                    d_['nu_TOPO'], d_['nu_LSRK'], d_['tstamp_ID'])

    # convert the LSRK frequency grid to a velocity grid
    v_LSRK = const.c_ * (1 - idata.nu_LSRK / inp.restfreq)

    ### Down-select to the desired velocity range
    # Treat potential velocity range input issues
    if vra is None:
        vra = [ivel[chpad], ivel[-chpad]]

        if np.diff(vra) < 0:
            vra = vra[::-1]

        if np.logical_and(vra[0] < ivel[chpad], vra[1] > ivel[-chpad]):
            print('The velocity range input is outside the data bounds:\n' + \
                  '    ...setting vra to conform to bounds and padding')
            vra = [ivel[chpad], ivel[-chpad]]

        if np.logical_and(vra[0] < ivel[chpad], vra[1] <= ivel[-chpad]):
            print('Low velocity range input is outside the data bounds: \n' + \
                  '    ...setting vra to conform to bounds and padding')
            vra[0] = ivel[chpad]

        if np.logical_and(vra[0] >= ivel[chpad], vra[1] > ivel[-chpad]):
            print('High velocity range input is outside the data bounds: \n' + \
                  '    ...setting vra to conform to bounds and padding')
            vra[1] = ivel[-chpad]

        # identify the channel indices that span the desired velocity range
        vral_ix = np.max(np.where(ivel <= vra[0]))
        vrah_ix = np.min(np.where(ivel >= vra[1]))

        # ensure this index range is evenly divisible by the binning factor
        # (we choose to add to the high-velocity end, but that's arbitrary)
        vrah_ix += (vrah_ix - vral_ix) % chbin


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


