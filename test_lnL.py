import os, sys, time
import numpy as np
from cube_parser import cube_parser
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from prep_data import *
import mconfig as in_
import const as c_


def lnprob(data, theta, theta_fixed, gcf_holder=None, corr_cache=None):

    # generate a model cube
    ntheta = len(theta)
    restfreq, FOV, Npix, dist, rmax = theta_fixed
    mcube = cube_parser(theta[:ntheta-3], FOV=FOV, Npix=Npix, dist=dist, 
                        r_max=rmax, Vsys=theta[10], restfreq=restfreq,
                        vel=data.vmod)

    # sample the FT of the cube onto the observed spatial frequencies
    if gcf_holder is None:
        mvis = vis_sample(imagefile=mcube, uu=data.u, vv=data.v, 
                          mu_RA=theta[11], mu_DEC=theta[12],
                          mod_interp=False).T
    else:
        mvis = vis_sample(imagefile=mcube, gcf_holder=gcf_holder, 
                          corr_cache=corr_cache, mod_interp=False, 
                          mu_RA=theta[11], mu_DEC=theta[12]).T

    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=0, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=0, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im

    # interpolation to native-resolution output channels
    fint = interp1d(data.vmod, mvis, axis=0, fill_value='extrapolate')
    mvis = fint(data.vobs)

    # spectral binning
    mwgt = np.ones_like(mvis.real)	# <--- unsure if this is wise
    chbin = np.int(mvis.shape[0] / len(data.vel))
    mvis_bin = np.average(mvis.reshape((-1, chbin, mvis.shape[1])),
                          weights=mwgt.reshape((-1, chbin, mwgt.shape[1])), 
                          axis=1)

    # compute the unnormalized log-likelihood for each polarization
    Minv = np.linalg.inv(data.cov)
    lnL = 0
    for i in range(data.vis.shape[0]):
        resid = np.absolute(data.vis[i,:,:] - mvis_bin)
        lnL -= 0.5 * np.tensordot(resid, np.dot(Minv, data.wgt[i,:,:] * resid))

    return lnL #+ data.lnL0



# load data
datafile = 'data/simp3_assigned_interp-short_medr_medv'
data = prep_data(datafile, vra=[-4e3, 12e3])

# load parameters
theta = in_.pars

# load fixed parameters
theta_fixed = in_.restfreq, in_.FOV, in_.Npix, in_.dist, in_.rmax

# calculate a log-likelihood
lnL = lnprob(data, theta, theta_fixed)
print(-2 * lnL)
