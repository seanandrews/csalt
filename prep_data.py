import os, sys
import numpy as np


def prep_data(file_prefix, vra=None, chbin=2, chpad=3, restfreq=230.538e9):

    ### Check inputs for consistency / rules
    # temporary
    if chbin != 2:
        print('do not be an asshole: set chbin = 2')
        chbin = 2

    if chbin >= chpad:
        print('chpad must be > chbin: setting chpad to %d' % chbin + 1)
        chpad = chbin + 1


    ### Load and parse the data
    # load the data dictionary 
    data_dict = np.load(file_prefix+'.npz')

    # loop through the execution blocks
    for i in range(data_dict['nobs']):

        # unpack the data
        vdata = data_dict[str(i)]
        nu_LSRK = vdata.nu_LSRK
        V_LSRK = const.c_ * (1 - nu_LSRK / restfreq)
        

    #iu, iv, ifreq, ivel = dat['u'], dat['v'], dat['freq'], dat['vel']
    #ivis, iwgt = dat['vis'], dat['weights']
    #vLSRK_grid = dat['vel_LSRK_grid']
    #npol, nvis = ivis.shape[0], ivis.shape[2]


    ### Down-select to the desired velocity range
    # treat potential velocity range input issues
    if vra is None:
        vra = [ivel[chpad], ivel[-chpad]]

    if np.diff(vra) < 0:
        vra = vra[::-1]

    if np.logical_and(vra[0] < ivel[chpad], vra[1] > ivel[-chpad]):
        print('The velocity range input is outside the data bounds: \n' + \
              '    ...setting vra to conform to bounds and padding')
        vra = [ivel[chpad], ivel[-chpad]]

    if np.logical_and(vra[0] < ivel[chpad], vra[1] <= ivel[-chpad]):
        print('The low velocity range input is outside the data bounds: \n' + \
              '    ...setting vra to conform to bounds and padding')
        vra[0] = ivel[chpad]

    if np.logical_and(vra[0] >= ivel[chpad], vra[1] > ivel[-chpad]):
        print('The high velocity range input is outside the data bounds: \n' + \
              '    ...setting vra to conform to bounds and padding')
        vra[1] = ivel[-chpad]

    # identify the subset of LSRK channels at the EB midpoint that overlap with
    # this range (+/- some padding), to pass to model calculation
    ivmod = vLSRK_grid[np.int(vLSRK_grid.shape[0] / 2),:]
    vmlo_ix = np.max(np.where(ivmod <= vra[0])) - chpad
    vmhi_ix = np.min(np.where(ivmod >= vra[1])) + chpad
    vmod = ivmod[vmlo_ix:vmhi_ix]

    # identify the subset of output (LSRK) channels in this range; ensure that 
    # this is evenly divisible by the binning factor
    vdlo_ix = np.max(np.where(ivel <= vra[0]))
    vdhi_ix = np.min(np.where(ivel >= vra[1]))
    vobs = ivel[vdlo_ix:vdhi_ix + ((vdhi_ix - vdlo_ix) % chbin)]

    # trim the data to these output (LSRK) channels only
    vis = ivis[:,vdlo_ix:vdhi_ix + ((vdhi_ix - vdlo_ix) % chbin),:]
    wgt = iwgt[:,vdlo_ix:vdhi_ix + ((vdhi_ix - vdlo_ix) % chbin),:]

    # bin by factor chbin (a weighted, decimated average [like CASA/SPLIT])
    avg_wgt = wgt.reshape((npol, -1, chbin, nvis))
    vis_bin = np.average(vis.reshape((npol, -1, chbin, nvis)), 
                         weights=avg_wgt, axis=2)
    wgt_bin = np.sum(avg_wgt, axis=2)
    vel_bin = np.average(vobs.reshape(-1, chbin), axis=1)
    Nch = len(vel_bin)


    ### Package everything we need into a data class and return it
    return vdata(iu, iv, vis_bin, wgt_bin, vel_bin, vobs, vmod, cov, lnL0)
