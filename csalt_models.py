"""
    csalt_models.py

    Usage: 
        - import modules

    Outputs:
	- various
"""
import os, sys
import numpy as np
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import scipy.constants as sc
from vis_sample.classes import *
from parametric_disk import *


def vismodel_full(pars, fixed, dataset, 
                  chpad=3, oversample=None, noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, npix, dist, rmax = fixed
    npars = len(pars)

    # Spatial frequencies to lambda units
    uu = dataset.um * np.mean(dataset.nu_TOPO) / sc.c
    vv = dataset.vm * np.mean(dataset.nu_TOPO) / sc.c

    # Pad the frequency arrays
    dnu_TOPO = np.diff(dataset.nu_TOPO)[0]
    nu_TOPO_s = dataset.nu_TOPO[0] + dnu_TOPO * np.arange(-chpad, 0, 1)
    nu_TOPO_f = dataset.nu_TOPO[-1] + dnu_TOPO * np.arange(1, chpad+1, 1)
    dataset.nu_TOPO = np.concatenate((nu_TOPO_s, dataset.nu_TOPO, nu_TOPO_f))

    dnu_LSRK = np.diff(dataset.nu_LSRK, axis=1)[:,0]
    nu_LSRK_s = (dataset.nu_LSRK[:,0])[:,None] + \
                dnu_LSRK[:,None] * np.arange(-chpad, 0, 1)[None,:]
    nu_LSRK_f = (dataset.nu_LSRK[:,-1])[:,None] + \
                dnu_LSRK[:,None] * np.arange(1, chpad+1, 1)[None,:]
    dataset.nu_LSRK = np.concatenate((nu_LSRK_s, dataset.nu_LSRK, nu_LSRK_f),
                                     axis=1)

    # Upsample in the spectral domain (if necessary)
    if oversample is not None:
        nchan = dataset.nchan + 2 * chpad
        nu_TOPO = np.interp(np.arange((nchan-1) * oversample + 1),
                            np.arange(0, nchan * oversample, oversample),
                            dataset.nu_TOPO)
        nch = len(nu_TOPO)
        nu_LSRK = np.empty((dataset.nstamps, nch))
        for itime in range(dataset.nstamps):
            nu_LSRK[itime,:] = np.interp(np.arange((nchan-1) * oversample + 1),
                               np.arange(0, nchan*oversample, oversample), 
                               dataset.nu_LSRK[itime,:])
    else:
        nu_TOPO = dataset.nu_TOPO
        nu_LSRK = dataset.nu_LSRK
        nch = len(nu_TOPO)
        oversample = 1

    # LSRK velocities 
    v_LSRK = sc.c * (1 - nu_LSRK / restfreq)


    ### - Configure noise (if necessary)
    if noise_inject is not None:
        # Scale input RMS for desired (naturally-weighted) noise per vis-chan
        sigma_out = 1e-3 * noise_inject * np.sqrt(dataset.npol * dataset.nvis)

        # Scale to account for spectral oversampling and SRF convolution
        sigma_noise = sigma_out * np.sqrt(np.pi * oversample)

        # Random Gaussian noise draws: note RE/IM separated for speed later
        noise = np.random.normal(0, sigma_noise, 
                                 (dataset.npol, nch, dataset.nvis, 2))
        noise = np.squeeze(noise)


    ### - Compute the model visibilities
    # Loop through timestamps to get raw (sky) visibilities
    mvis_pure = np.squeeze(np.empty((dataset.npol, nch, dataset.nvis, 2)))
    for itime in range(dataset.nstamps):
        # track the steps
        print('timestamp '+str(itime+1)+' / '+str(dataset.nstamps))

        # create a model cube
        cube = parametric_disk(v_LSRK[itime,:], pars, fixed)

        # indices for this timestamp only
        ixl = np.min(np.where(dataset.tstamp == itime))
        ixh = np.max(np.where(dataset.tstamp == itime)) + 1

        # sample it's Fourier transform on the template (u,v) spacings
        mvis = vis_sample(imagefile=cube, uu=uu[ixl:ixh], vv=vv[ixl:ixh],
                          mu_RA=pars[11], mu_DEC=pars[12], mod_interp=False).T

        # populate the results in the output array *for this timestamp only*
        mvis_pure[0,:,ixl:ixh,0] = mvis.real
        mvis_pure[1,:,ixl:ixh,0] = mvis.real
        mvis_pure[0,:,ixl:ixh,1] = mvis.imag
        mvis_pure[1,:,ixl:ixh,1] = mvis.imag


    # Convolve with the spectral response function
    # (truncated at a 40 dB power drop for computational speed)
    chix = np.arange(25 * oversample) / oversample
    xch = chix - np.mean(chix)
    SRF = 0.5 * np.sinc(xch) + 0.25 * np.sinc(xch-1) + 0.25 * np.sinc(xch+1)
    mvis_pure = convolve1d(mvis_pure, SRF/np.sum(SRF), axis=1, mode='nearest')

    # Return decimated visibilities, with noise if necessary
    if noise_inject is None:
        # Decimate and remove padding
        mvis_pure = mvis_pure[:,::oversample,:,:]
        mvis_pure = mvis_pure[:,chpad:-chpad,:,:]

        # Convert to complex and return
        return mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
    else:
        # SRF convolution of noisy data
        mvis_noisy = convolve1d(mvis_pure + noise, SRF/np.sum(SRF), 
                                axis=1, mode='nearest')

        # Decimate
        mvis_pure = mvis_pure[:,::oversample,:,:]
        mvis_pure = mvis_pure[:,chpad:-chpad,:,:]
        mvis_noisy = mvis_noisy[:,::oversample,:,:]
        mvis_noisy = mvis_noisy[:,chpad:-chpad,:,:]

        # Convert to complex
        mvis_pure = mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
        mvis_noisy = mvis_noisy[:,:,:,0] + 1j * mvis_noisy[:,:,:,1]

        return mvis_pure, mvis_noisy






def vismodel_def(pars, fixed, dataset, 
                 imethod='cubic', return_holders=False, chpad=3):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, rmax = fixed
    npars = len(pars)

    # Spatial frequencies to lambda units
    uu = dataset.um * np.mean(dataset.nu_TOPO) / sc.c
    vv = dataset.vm * np.mean(dataset.nu_TOPO) / sc.c

    # Pad the frequency arrays
    dnu_TOPO = np.diff(dataset.nu_TOPO)[0]
    nu_TOPO_s = dataset.nu_TOPO[0] + dnu_TOPO * np.arange(-chpad, 0, 1)
    nu_TOPO_f = dataset.nu_TOPO[-1] + dnu_TOPO * np.arange(1, chpad+1, 1)
    nu_TOPO = np.concatenate((nu_TOPO_s, dataset.nu_TOPO, nu_TOPO_f))

    dnu_LSRK = np.diff(dataset.nu_LSRK, axis=1)[:,0]
    nu_LSRK_s = (dataset.nu_LSRK[:,0])[:,None] + \
                dnu_LSRK[:,None] * np.arange(-chpad, 0, 1)[None,:]
    nu_LSRK_f = (dataset.nu_LSRK[:,-1])[:,None] + \
                dnu_LSRK[:,None] * np.arange(1, chpad+1, 1)[None,:]
    nu_LSRK = np.concatenate((nu_LSRK_s, dataset.nu_LSRK, nu_LSRK_f), axis=1)

    # LSRK velocities at midpoint of execution block
    mid_stamp = np.int(nu_LSRK.shape[0] / 2)
    v_model = sc.c * (1 - nu_LSRK[mid_stamp,:] / restfreq)
    v_grid = sc.c * (1 - nu_LSRK / restfreq)

    # generate a model cube
    mcube = parametric_disk(v_model, pars, fixed)

    # sample the FT of the cube onto the observed spatial frequencies
    mvis, gcf, corr = vis_sample(imagefile=mcube, uu=uu, vv=vv, mu_RA=pars[11], 
                                 mu_DEC=pars[12], return_gcf=True, 
                                 return_corr_cache=True, mod_interp=False)
    mvis = mvis.T

    # distribute interpolates to different timestamps
    for itime in range(dataset.nstamps):
        ixl = np.min(np.where(dataset.tstamp == itime))
        ixh = np.max(np.where(dataset.tstamp == itime)) + 1
        fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind=imethod, 
                        fill_value='extrapolate')
        mvis[:,ixl:ixh] = fint(v_grid[itime,:])

    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=0, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=0, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im
    mvis = mvis[chpad:-chpad,:]

    # populate both polarizations
    mvis = np.tile(mvis, (2, 1, 1))

    # return the dataset after replacing the visibilities with the model
    if return_holders:
        return mvis, gcf, corr
    else:
        return mvis


def vismodel_iter(pars, fixed, dataset, gcf, corr, imethod='cubic', chpad=3):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, rmax = fixed
    npars = len(pars)

    # Pad the frequency arrays
    dnu_TOPO = np.diff(dataset.nu_TOPO)[0]
    nu_TOPO_s = dataset.nu_TOPO[0] + dnu_TOPO * np.arange(-chpad, 0, 1)
    nu_TOPO_f = dataset.nu_TOPO[-1] + dnu_TOPO * np.arange(1, chpad+1, 1)
    nu_TOPO = np.concatenate((nu_TOPO_s, dataset.nu_TOPO, nu_TOPO_f))

    dnu_LSRK = np.diff(dataset.nu_LSRK, axis=1)[:,0]
    nu_LSRK_s = (dataset.nu_LSRK[:,0])[:,None] + \
                dnu_LSRK[:,None] * np.arange(-chpad, 0, 1)[None,:]
    nu_LSRK_f = (dataset.nu_LSRK[:,-1])[:,None] + \
                dnu_LSRK[:,None] * np.arange(1, chpad+1, 1)[None,:]
    nu_LSRK = np.concatenate((nu_LSRK_s, dataset.nu_LSRK, nu_LSRK_f), axis=1)

    # LSRK velocities at midpoint of execution block
    mid_stamp = np.int(dataset.nu_LSRK.shape[0] / 2)
    v_model = sc.c * (1 - nu_LSRK[mid_stamp,:] / restfreq)
    v_grid = sc.c * (1 - nu_LSRK / restfreq)

    # generate a model cube
    mcube = parametric_disk(v_model, pars, fixed)

    # sample the FT of the cube onto the observed spatial frequencies
    mvis = vis_sample(imagefile=mcube, mu_RA=pars[11], mu_DEC=pars[12], 
                      gcf_holder=gcf, corr_cache=corr, mod_interp=False).T

    # distribute interpolates to different timestamps
    for itime in range(dataset.nstamps):
        ixl = np.min(np.where(dataset.tstamp == itime))
        ixh = np.max(np.where(dataset.tstamp == itime)) + 1
        fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind=imethod,
                        fill_value='extrapolate')
        mvis[:,ixl:ixh] = fint(v_grid[itime,:])

    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis_re = convolve1d(mvis.real, SRF_kernel, axis=0, mode='nearest')
    mvis_im = convolve1d(mvis.imag, SRF_kernel, axis=0, mode='nearest')
    mvis = mvis_re + 1.0j*mvis_im
    mvis = mvis[chpad:-chpad,:]

    # populate both polarizations
    mvis = np.tile(mvis, (2, 1, 1))

    # return the model visibilities
    return mvis
