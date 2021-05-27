import os, sys
import numpy as np


class vdata:
   def __init__(vis, wgt, vel, vobs, vmod, cov, cov_inv, lnL0):
        self.vis = vis
        self.wgt = wgt
        self.vel = vel
        self.vobs = vobs
        self.vmod = vmod
        self.cov = cov
        self.cov_inv = cov_inv
        self.lnL0 = lnL0


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
    # load from storage
    dat = np.load(file_prefix+'.npz')
    iu, iv, ifreq, ivel = dat['u'], dat['v'], dat['freq'], dat['vel']
    ivis, iwgt = dat['vis'], dat['weights']
    vLSRK_grid = dat['vel_LSRK_grid']
    npol, nvis = ivis.shape[0], ivis.shape[2]

    # if necessary, convert to spectral-dependent weights
    if (iwgt.shape != ivis.shape):
        iwgt = np.rollaxis(np.tile(iwgt, (len(ivel), 1, 1)), 1)

    # if necessary, swap so channel velocities are increasing
    if np.any(np.diff(ivel) < 0):
        ifreq, ivel = ifreq[::-1], ivel[::-1]
        ivis, iwgt = ivis[:,::-1,:], iwgt[:,::-1,:]

    if np.any(np.diff(vLSRK_grid, axis=1) < 0):
        vLSRK_grid = vLSRK_grid[:,::-1]


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


    ### Pre-calculate important inference quantities
    # covariance matrix and its inverse
    cov = (5./16)*np.eye(Nch) + (3./32)*(np.eye(Nch, k=-1) + np.eye(Nch, k=1))
    cov_inv = np.linalg.inv(cov)

    # log-likelihood normalization
    dterm = np.empty((npol, nvis))
    for i in range(nvis):
        for j in range(npol):
            sgn, lndet = np.linalg.slogdet(cov / wgt_bin[j,:,i])
            dterm[j,i] = sgn * lndet
    lnL0 = -0.5 * (np.prod(vis_bin.shape) * np.log(2 * np.pi) + np.sum(dterm))


    ### Package everything we need into a data class and return it
    return vdata(vis_bin, wgt_bin, vel_bin, vobs, vmod, cov, cov_inv, lnL0)   
