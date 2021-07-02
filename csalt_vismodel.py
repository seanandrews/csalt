import os, sys
import numpy as np
from cube_parser import cube_parser
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
import const as const
import matplotlib.pyplot as plt


def csalt_vismodel(dataset, theta, theta_fixed, return_holders=False):

    # parse the parameters
    ntheta = len(theta)
    restfreq, FOV, Npix, dist, rmax = theta_fixed

    # parse the velocities: in this case, use the LSRK velocities at the 
    # midpoint of the execution block
    mid_stamp = dataset.nu_LSRK.shape[0] / 2
    if (mid_stamp % 1) != 0:
        ilo = np.int(mid_stamp + mid_stamp % 1)
        nu_midpt = np.mean(dataset.nu_LSRK[ilo-1:ilo+1,:], axis=0)
    else:
        nu_midpt = dataset.nu_LSRK[np.int(mid_stamp),:]
    v_model = const.c_ * (1 - nu_midpt / restfreq)

    # generate a model cube
    mcube = cube_parser(theta[:ntheta-3], FOV=FOV, Npix=Npix, dist=dist, 
                        r_max=rmax, Vsys=theta[10], restfreq=restfreq, 
                        vel=v_model)

    # sample the FT of the cube onto the observed spatial frequencies
    if return_holders:
        mvis, gcf, corr = vis_sample(imagefile=mcube, 
                                     uu=dataset.u, vv=dataset.v,
                                     mu_RA=theta[11], mu_DEC=theta[12],
                                     return_gcf=True, return_corr_cache=True,
                                     mod_interp=False)
    else:
        mvis = vis_sample(imagefile=mcube, uu=dataset.u, vv=dataset.v, 
                          mu_RA=theta[11], mu_DEC=theta[12], mod_interp=False)
    mvis = mvis.T

    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=0, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=0, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im

    # populate both polarizations
    mvis = np.tile(mvis, (2, 1, 1))

    # return the dataset after replacing the visibilities with the model
    if return_holders:
        return mvis, gcf, corr
    else:
        return mvis


def csalt_vismodel_iter(dataset, theta, theta_fixed, v_model, gcf, corr):

    # parse the parameters
    ntheta = len(theta)
    restfreq, FOV, Npix, dist, rmax = theta_fixed

    # generate a model cube
    mcube = cube_parser(theta[:ntheta-3], FOV=FOV, Npix=Npix, dist=dist,
                        r_max=rmax, Vsys=theta[10], restfreq=restfreq, 
                        vel=v_model)

    # sample the FT of the cube onto the observed spatial frequencies
    mvis = vis_sample(imagefile=mcube, mu_RA=theta[11], mu_DEC=theta[12], 
                      gcf_holder=gcf, corr_cache=corr, mod_interp=False).T

    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=0, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=0, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im

    # return the model visibilities
    return mvis
