import os, sys, time
import numpy as np
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from cube_parser import cube_parser
import matplotlib.pyplot as plt

def lnprob(data, theta, theta_fixed, method='quick'):

    # parse parameters
    ntheta = len(theta)
    restfreq, FOV, Npix, dist, rmax = theta_fixed

    if method == 'quick':
        # generate a model cube
        mcube = cube_parser(theta[:ntheta-3], FOV=FOV, Npix=Npix, dist=dist, 
                            r_max=rmax, Vsys=theta[10], restfreq=restfreq,
                            vel=data.vmod)

        # sample the FT of the cube onto the observed spatial frequencies
        mvis = vis_sample(imagefile=mcube, uu=data.u, vv=data.v, 
                          mu_RA=theta[11], mu_DEC=theta[12], mod_interp=False).T


    if method == 'interp':
        mcube = cube_parser(theta[:ntheta-3], FOV=FOV, Npix=Npix, dist=dist,
                            r_max=rmax, Vsys=theta[10], restfreq=restfreq,
                            vel=data.vmod)

        mvisi = vis_sample(imagefile=mcube, uu=data.u, vv=data.v,
                           mu_RA=theta[11], mu_DEC=theta[12], 
                           mod_interp=False).T

        nstamps = data.vmod_grid.shape[0]
        nperstamp = np.int(data.vis.shape[-1] / nstamps)
        mvis = np.empty_like(mvisi)
        for i in range(nstamps):
            ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
            fint = interp1d(data.vmod, mvisi[:,ix_lo:ix_hi], axis=0, 
                            kind='cubic', fill_value='extrapolate')
            mvis[:,ix_lo:ix_hi] = fint(data.vmod_grid[i,:])


    if method == 'full':
        nstamps = data.vmod_grid.shape[0]
        nperstamp = np.int(data.vis.shape[-1] / nstamps)
        for i in range(nstamps):
            mcube = cube_parser(theta[:ntheta-3], FOV=FOV, Npix=Npix, 
                                dist=dist, r_max=rmax, Vsys=theta[10], 
                                restfreq=restfreq, vel=data.vmod_grid[i,:])

            mvisi = vis_sample(imagefile=mcube, uu=data.u, vv=data.v, 
                               mu_RA=theta[11], mu_DEC=theta[12],
                               mod_interp=False).T
            if i == 0: mvis = mvisi.copy()
            ix_lo, ix_hi = i * nperstamp, (i + 1) * nperstamp
            mvis[:,ix_lo:ix_hi] = mvisi[:,ix_lo:ix_hi]



    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=0, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=0, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im

    # trim off pad channels
    mvis = mvis[3:-3,:]

    # spectral binning
    mwgt = np.ones_like(mvis.real)	# <--- unsure if this is wise
    chbin = 2	#np.int(mvis.shape[0] / len(data.vel))
    mvis_bin = np.average(mvis.reshape((-1, chbin, mvis.shape[1])),
                          weights=mwgt.reshape((-1, chbin, mwgt.shape[1])), 
                          axis=1)

    # compute the unnormalized log-likelihood for each polarization
    Minv = np.linalg.inv(data.cov)
    lnL = 0
    for i in range(data.vis.shape[0]):
        resid = np.absolute(data.vis_bin[i,:,:] - mvis_bin)
        lnL -= 0.5*np.tensordot(resid, np.dot(Minv, data.wgt_bin[i,:,:]*resid))

    return lnL #+ data.lnL0



import importlib
from prep_data import prep_data

# load inputs
inp = importlib.import_module('fconfig_simp3-nmm')

# load data
ddict = prep_data(inp.dataname, vra=[0.4e3, 10e3])
data_EB0 = ddict['0']


# load parameters
theta = np.array([inp.incl, inp.PA, inp.mstar, inp.r_l, inp.z0, inp.psi,
                  inp.Tb0, inp.q, inp.Tback, inp.dV0,
                  inp.vsys, inp.xoff, inp.yoff])
theta_fixed = inp.nu_rest, inp.FOV, inp.Npix, inp.dist, inp.rmax

# calculate a log-likelihood
t0 = time.time()
chi2_quick = -2 * lnprob(data_EB0, theta, theta_fixed, method='quick')
t1 = time.time()
chi2_interp = -2 * lnprob(data_EB0, theta, theta_fixed, method='interp')
t2 = time.time()
#chi2_full = -2 * lnprob(data_EB0, theta, theta_fixed, method='full')
t3 = time.time()

print(chi2_interp, chi2_quick)
print(t2-t1, t1-t0)
