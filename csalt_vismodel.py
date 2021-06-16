import os, sys
import numpy as np
from cube_parser import cube_parser
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
import const as const


def csalt_vismodel_single(dataset, theta, theta_fixed, return_holders=False):

    # parse the parameters
    ntheta = len(theta)
    restfreq, FOV, Npix, dist, rmax = theta_fixed

    # parse the velocities: in this case, use the LSRK velocities at the 
    # midpoint of the execution block
    mid_stamp = dataset.nu_LSRK.shape[0] / 2
    if (mid_stamp % 1) != 0:
        ilo = np.int(mid_stamp + mid_stamp % 1)
        nu_midpt = np.average(dataset.nu_LSRK[ilo-1:ilo+1,:])
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

    # return the dataset after replacing the visibilities with the model
    if return_holders:
        return mvis, gcf, corr
    else:
        return mvis




class vdata:
   def __init__(self, u, v, vis, wgt, nu_topo, nu_lsrk):
        self.u = u
        self.v = v
        self.vis = vis
        self.wgt = wgt
        self.nu_TOPO = nu_topo
        self.nu_LSRK = nu_lsrk

# load data
data_dict = np.load('data/Sz129/Sz129.npy', allow_pickle=True).item()
nobs = data_dict['nobs']

d0 = np.load('data/Sz129/Sz129_EB0.npz')
dataset0 = vdata(d0['u'], d0['v'], d0['data'], d0['weights'], 
                 d0['nu_TOPO'], d0['nu_LSRK'])

# parameters
theta = np.array([40, 130, 0.7, 200, 2.3, 1, 205, 0.5, 20, 348, 5.2e3, 0, 0])
theta_fixed = 230.538e9, 8.0, 256, 150, 10

modelvis = csalt_vismodel_single(dataset0, theta, theta_fixed)

print(dataset0.vis.shape, modelvis.shape)
