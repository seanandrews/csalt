"""
    csalt_models.py

    Usage: 
        - import modules

    Outputs:
	- various
"""
import os, sys, importlib, time
import numpy as np
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
import scipy.constants as sc
import matplotlib.pyplot as plt
from vis_sample.classes import *
from astropy.io import fits, ascii
from csalt.parametric_disk_CSALT import parametric_disk as par_disk_CSALT
from csalt.parametric_disk_FITS import parametric_disk as par_disk_FITS
from csalt.parametric_disk_MCFOST import parametric_disk as par_disk_MCFOST

def cube_to_fits(sky_image, fitsout, RA=0., DEC=0., restfreq=230.538e9):

    # revert to proper formatting
    cube = np.rollaxis(np.fliplr(sky_image.data), -1)

    # extract coordinate information
    im_nfreq, im_ny, im_nx = cube.shape
    pixsize_x = np.abs(np.diff(sky_image.ra)[0])
    pixsize_y = np.abs(np.diff(sky_image.dec)[0])
    CRVAL3 = sky_image.freqs[0]
    if len(sky_image.freqs) > 1:
        CDELT3 = np.diff(sky_image.freqs)[0]
    else:
        CDELT3 = 1

    # generate the primary HDU
    hdu = fits.PrimaryHDU(np.float32(cube))

    # generate the header
    header = hdu.header
    header['EPOCH'] = 2000.
    header['EQUINOX'] = 2000.

    # Latitude and Longitude of the pole of the coordinate system.
    header['LATPOLE'] = -1.436915713634E+01
    header['LONPOLE'] = 180.

    # Define the RA coordinate
    header['CTYPE1'] = 'RA---SIN'
    header['CUNIT1'] = 'DEG'
    header['CDELT1'] = -pixsize_x / 3600 
    header['CRPIX1'] = 0.5 * im_nx + 0.5
    header['CRVAL1'] = RA

    # Define the DEC coordinate
    header['CTYPE2'] = 'DEC--SIN'
    header['CUNIT2'] = 'DEG'
    header['CDELT2'] = pixsize_y / 3600 
    header['CRPIX2'] = 0.5 * im_ny + 0.5
    header['CRVAL2'] = DEC

    # Define the frequency coordiante
    header['CTYPE3'] = 'FREQ'
    header['CUNIT3'] = 'Hz'
    header['CRPIX3'] = 1.
    header['CDELT3'] = CDELT3
    header['CRVAL3'] = CRVAL3

    header['SPECSYS'] = 'LSRK'
    header['VELREF'] = 257
    header['RESTFREQ'] = restfreq
    header['BSCALE'] = 1.
    header['BZERO'] = 0.
    header['BUNIT'] = 'JY/PIXEL'
    header['BTYPE'] = 'Intensity'

    hdu.writeto(fitsout, overwrite=True)

    return


def radmc_to_fits(path_to_image, fitsout, pars_fixed):

    # parse fixed parameters
    restfreq, FOV, npix, dist, cfg_dict = pars_fixed

    # load the output into a proper cube array
    imagefile = open(path_to_image+'/image.out')
    iformat = imagefile.readline()
    im_nx, im_ny = imagefile.readline().split() #npixels along x and y axes
    im_nx, im_ny = np.int(im_nx), np.int(im_ny)
    nlam = np.int(imagefile.readline())

    pixsize_x, pixsize_y = imagefile.readline().split() #pixel sizes in cm 
    pixsize_x = np.float(pixsize_x)
    pixsize_y = np.float(pixsize_y)

    imvals = ascii.read(path_to_image+'/image.out', format='fast_csv',
                        guess=False, data_start=4,
                        fast_reader={'use_fast_converter':True})['1']
    lams = imvals[:nlam]

    # erg cm^-2 s^-1 Hz^-1 str^-1 --> Jy / pixel
    cube = np.reshape(imvals[nlam:],[nlam, im_ny, im_nx])
    cube *= 1e23 * pixsize_x * pixsize_y / (dist * 3.0857e18)**2

    # Pack the cube into a vis_sample SkyImage object and FITS file
    mod_data = np.rollaxis(cube, 0, 3)
    mod_ra  = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix)
    mod_dec = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix)
    freq = sc.c / (lams * 1e-6)

    skyim = SkyImage(mod_data, mod_ra, mod_dec, freq, None)
    foo = cube_to_fits(skyim, fitsout, 240., -30., restfreq=restfreq)

    return 
    



def vismodel_full(pars, fixed, dataset, mtype='CSALT',
                  chpad=3, oversample=None, noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, npix, dist, cfg_dict = fixed
    npars = len(pars)

    # Load appropriate model for cube calculation
    pd = importlib.import_module('parametric_disk_'+mtype)

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
        sigma_noise = sigma_out * np.sqrt(oversample * 8./3.)

        # Random Gaussian noise draws: note RE/IM separated for speed later
        noise = np.random.normal(0, sigma_noise, 
                                 (dataset.npol, nch, dataset.nvis, 2))
        noise = np.squeeze(noise)


    ### - Compute the model visibilities
    # Loop through timestamps to get raw (sky) visibilities
    mvis_ = np.squeeze(np.empty((dataset.npol, nch, dataset.nvis, 2)))
    t0 = time.time()
    for itime in range(dataset.nstamps):
        # track the steps
        print('timestamp '+str(itime+1)+' / '+str(dataset.nstamps))

        # create a model cube
        cube = pd.parametric_disk(v_LSRK[itime,:], pars, fixed)

        # indices for this timestamp only
        ixl = np.min(np.where(dataset.tstamp == itime))
        ixh = np.max(np.where(dataset.tstamp == itime)) + 1

        # sample it's Fourier transform on the template (u,v) spacings
        mvis = vis_sample(imagefile=cube, uu=uu[ixl:ixh], vv=vv[ixl:ixh],
                          mu_RA=pars[-2], mu_DEC=pars[-1], mod_interp=False).T

        # populate the results in the output array *for this timestamp only*
        mvis_[0,:,ixl:ixh,0] = mvis.real
        mvis_[1,:,ixl:ixh,0] = mvis.real
        mvis_[0,:,ixl:ixh,1] = mvis.imag
        mvis_[1,:,ixl:ixh,1] = mvis.imag
    print('time to make this == ', (time.time()-t0) / 60.)

    # Convolve with the spectral response function
    # (truncated at a 40 dB power drop for computational speed)
    if oversample > 1:
        chix = np.arange(25 * oversample) / oversample
        xch = chix - np.mean(chix)
        SRF = 0.5*np.sinc(xch) + 0.25*np.sinc(xch-1) + 0.25*np.sinc(xch+1)
    else:
        SRF = np.array([0.0, 0.25, 0.50, 0.25, 0.0])
    mvis_pure = convolve1d(mvis_, SRF/np.sum(SRF), axis=1, mode='nearest')
#    mvis_pure = 1. * mvis_

    # Return decimated visibilities, with noise if necessary
    if noise_inject is None:
        # Decimate and remove padding
        mvis_pure = mvis_pure[:,::oversample,:,:]
        mvis_pure = mvis_pure[:,chpad:-chpad,:,:]

        # Convert to complex and return
        return mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
    else:
        # SRF convolution of noisy data
        mvis_noisy = convolve1d(mvis_ + noise, SRF/np.sum(SRF), 
                                axis=1, mode='nearest')
        #mvis_noisy = mvis_ + noise

        # Decimate
        mvis_pure = mvis_pure[:,::oversample,:,:]
        mvis_pure = mvis_pure[:,chpad:-chpad,:,:]
        mvis_noisy = mvis_noisy[:,::oversample,:,:]
        mvis_noisy = mvis_noisy[:,chpad:-chpad,:,:]

        # Convert to complex
        mvis_pure = mvis_pure[:,:,:,0] + 1j * mvis_pure[:,:,:,1]
        mvis_noisy = mvis_noisy[:,:,:,0] + 1j * mvis_noisy[:,:,:,1]

        return mvis_pure, mvis_noisy




def vismodel_def(pars, fixed, dataset, mtype='CSALT',
                 imethod='cubic', return_holders=False, chpad=3,
                 redo_RTimage=True, noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, cfg_dict = fixed
    npars = len(pars)

    # Load appropriate model for cube calculation
    pd = importlib.import_module('parametric_disk_'+mtype)

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
    mcube = pd.parametric_disk(v_model, pars, fixed, newcube=redo_RTimage)

    # sample the FT of the cube onto the observed spatial frequencies
    mvis_, gcf, corr = vis_sample(imagefile=mcube, uu=uu, vv=vv,
                                  mu_RA=pars[-2], mu_DEC=pars[-1],
                                  return_gcf=True, return_corr_cache=True, 
                                  mod_interp=False)
    mvis_ = mvis_.T

    # distribute interpolates to different timestamps
    for itime in range(dataset.nstamps):
        ixl = np.min(np.where(dataset.tstamp == itime))
        ixh = np.max(np.where(dataset.tstamp == itime)) + 1
        fint = interp1d(v_model, mvis_[:,ixl:ixh], axis=0, kind=imethod, 
                        fill_value='extrapolate')
        mvis_[:,ixl:ixh] = fint(v_grid[itime,:])

    ### - Configure noise (if necessary)
    if noise_inject is not None:
        # Scale input RMS for desired (naturally-weighted) noise per vis-chan
        sigma_out = 1e-3 * noise_inject * np.sqrt(dataset.npol * dataset.nvis)
        sigma_noise = sigma_out * 1.16 #* np.sqrt(8./3.)

        # Random Gaussian noise draws
        noise = np.random.normal(0, sigma_noise, 
                                 (mvis_.shape[0], dataset.nvis, 2))
        noise = np.squeeze(noise)
        noise = noise[:,:,0] + 1j * noise[:,:,1]
        
    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    p_mvis = convolve1d(mvis_.real, SRF_kernel, axis=0, mode='nearest') + \
             1j*convolve1d(mvis_.imag, SRF_kernel, axis=0, mode='nearest')

    # return the dataset after replacing the visibilities with the model
    if return_holders:
        # remove pads
        p_mvis = p_mvis[chpad:-chpad,:]
        
        # populate both polarizations
        p_mvis = np.tile(p_mvis, (2, 1, 1))

        return p_mvis, gcf, corr

    elif noise_inject is not None:
        # inject noise before SRF convolution 
        mvis_ += noise
        n_mvis = convolve1d(mvis_.real, SRF_kernel, axis=0, mode='nearest') + \
                 1j*convolve1d(mvis_.imag, SRF_kernel, axis=0, mode='nearest')

        # remove pads
        p_mvis = p_mvis[chpad:-chpad,:]
        n_mvis = n_mvis[chpad:-chpad,:]
      
        # populate both polarizations
        p_mvis = np.tile(p_mvis, (2, 1, 1))
        n_mvis = np.tile(n_mvis, (2, 1, 1))

        return p_mvis, n_mvis
        
    else:
        # remove pads
        p_mvis = p_mvis[chpad:-chpad,:]
        
        # populate both polarizations
        p_mvis = np.tile(p_mvis, (2, 1, 1))

        return p_mvis


def vismodel_iter(pars, fixed, dataset, gcf, corr, imethod='cubic', chpad=3, code='default'):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, cfg_dict = fixed
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
    if code == 'mcfost':
        mcube = par_disk_MCFOST(v_model, pars, fixed)
    else:
        mcube = par_disk_CSALT(v_model, pars, fixed)

    # sample the FT of the cube onto the observed spatial frequencies
    mvis = vis_sample(imagefile=mcube, mu_RA=pars[-2], mu_DEC=pars[-1], 
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



def vismodel_naif(pars, fixed, dataset, gcf=None, corr=None, mtype='CSALT',
                  return_holders=False, redo_RTimage=True, noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, cfg_dict = fixed
    npars = len(pars)

    # LSRK velocities 
    v_model = sc.c * (1 - dataset.nu_LSRK[0,:] / restfreq)

    # generate a model cube
#    mcube = par_disk_CSALT(v_model, pars, fixed)
    # Load appropriate model for cube calculation
    pd = importlib.import_module('parametric_disk_'+mtype)
    mcube = pd.parametric_disk(v_model, pars, fixed, newcube=redo_RTimage)

    # sample the FT of the cube onto the observed spatial frequencies
    if return_holders:
        uu = dataset.um * np.mean(dataset.nu_TOPO) / sc.c
        vv = dataset.vm * np.mean(dataset.nu_TOPO) / sc.c
        mvis_, gcf, corr = vis_sample(imagefile=mcube, uu=uu, vv=vv,
                                      mu_RA=pars[-2], mu_DEC=pars[-1],
                                      return_gcf=True, return_corr_cache=True,
                                      mod_interp=False)
        mvis = mvis_.T
        return mvis, gcf, corr
    else:
        mvis = vis_sample(imagefile=mcube, mu_RA=pars[-2], mu_DEC=pars[-1],
                          gcf_holder=gcf, corr_cache=corr, mod_interp=False).T

        # populate both polarizations
        mvis = np.tile(mvis, (2, 1, 1))

        # return the model visibilities
        return mvis



def vismodel_naif_wdoppcorr(pars, fixed, dataset, gcf=None, corr=None,
                            return_holders=False, imethod='cubic',
                            mtype='CSALT', redo_RTimage=True, noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, cfg_dict = fixed
    npars = len(pars)

    # LSRK velocities 
    mid_stamp = np.int(dataset.nu_LSRK.shape[0] / 2)
    v_model = sc.c * (1 - dataset.nu_LSRK[mid_stamp,:] / restfreq)
    v_grid = sc.c * (1 - dataset.nu_LSRK / restfreq)

    # generate a model cube
    #mcube = par_disk_CSALT(v_model, pars, fixed)
    pd = importlib.import_module('parametric_disk_'+mtype)
    mcube = pd.parametric_disk(v_model, pars, fixed, newcube=redo_RTimage)

    # sample the FT of the cube onto the observed spatial frequencies
    if return_holders:
        uu = dataset.um * np.mean(dataset.nu_TOPO) / sc.c
        vv = dataset.vm * np.mean(dataset.nu_TOPO) / sc.c
        mvis_, gcf, corr = vis_sample(imagefile=mcube, uu=uu, vv=vv,
                                      mu_RA=pars[-2], mu_DEC=pars[-1],
                                      return_gcf=True, return_corr_cache=True,
                                      mod_interp=False)
        mvis = mvis_.T
        return mvis, gcf, corr
    else:
        mvis = vis_sample(imagefile=mcube, mu_RA=pars[-2], mu_DEC=pars[-1],
                          gcf_holder=gcf, corr_cache=corr, mod_interp=False).T

        # Doppler correction
        for itime in range(dataset.nstamps):
            ixl = np.min(np.where(dataset.tstamp == itime))
            ixh = np.max(np.where(dataset.tstamp == itime)) + 1
            fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind=imethod,
                            fill_value='extrapolate')
            mvis[:,ixl:ixh] = fint(v_grid[itime,:])

        # populate both polarizations
        mvis = np.tile(mvis, (2, 1, 1))

        # return the model visibilities
        return mvis, mvis




def vismodel_FITS(pars, fixed, dataset,
                  imethod='cubic', return_holders=False, chpad=3,
                  noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, cfg_dict = fixed
    npars = len(pars)

    # Ingest the model cube 
    mcube = par_disk_FITS([0], pars, fixed)

    # Calculate the model cube velocities
    v_model = sc.c * (1 - mcube.freqs / restfreq)


    ### "Reduce" the input dataset to overlapping spectral coverage
    # Find the frequency axis indices nearest the min/max of the input cube
    chslo = np.argmin(np.abs(dataset.nu_LSRK - mcube.freqs.min()), axis=1)
    chshi = np.argmin(np.abs(dataset.nu_LSRK - mcube.freqs.max()), axis=1)

    # Determine the directionality of the frequency axis
    dir_spec = np.sign(np.diff(dataset.nu_TOPO)[0])

    # Select the channel index boundaries in the dataset, interior to the 
    # specified cube boundaries (to minimize passing problematic results due to
    # SRF convolution near the boundaries)
    ncut = 1
    if dir_spec > 0:
        chlo, chhi = np.max(chslo) + ncut, np.min(chshi) - ncut
    else:
        chhi, chlo = np.min(chslo) - ncut, np.max(chshi) + ncut

    # Extract only the spectral region of interest from the dataset
    dataset.nu_TOPO = dataset.nu_TOPO[chlo:chhi+1]
    dataset.nu_LSRK = dataset.nu_LSRK[:,chlo:chhi+1]
    dataset.vis = dataset.vis[:,chlo:chhi+1,:]
    dataset.nchan = len(dataset.nu_TOPO)
    

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

    # LSRK velocities 
    v_grid = sc.c * (1 - nu_LSRK / restfreq)

    # sample the FT of the cube onto the observed spatial frequencies
    mvis, gcf, corr = vis_sample(imagefile=mcube, uu=uu, vv=vv,
                                 return_gcf=True, return_corr_cache=True,
                                 mod_interp=False)
    mvis = mvis.T

    # distribute interpolates to different timestamps
    mvis_ = np.zeros((nu_LSRK.shape[1], mvis.shape[1]), dtype=complex)
    for itime in range(dataset.nstamps):
        ixl = np.min(np.where(dataset.tstamp == itime))
        ixh = np.max(np.where(dataset.tstamp == itime)) + 1
        fint = interp1d(v_model, mvis[:,ixl:ixh], axis=0, kind=imethod,
                        bounds_error=False, fill_value=0.0)
        mvis_[:,ixl:ixh] = fint(v_grid[itime,:])


    ### - Configure noise (if necessary)
    if noise_inject is not None:
        # Scale input RMS for desired (naturally-weighted) noise per vis-chan
        sigma_out = 1e-3 * noise_inject * np.sqrt(dataset.npol * dataset.nvis)
        sigma_noise = sigma_out * np.sqrt(8./3.)

        # Random Gaussian noise draws
        noise = np.random.normal(0, sigma_noise,
                                 (mvis_.shape[0], dataset.nvis, 2))
        noise = np.squeeze(noise)
        noise = noise[:,:,0] + 1j * noise[:,:,1]

    # convolve with the SRF
    SRF_kernel = np.array([0, 0.25, 0.5, 0.25, 0])
    p_mvis = convolve1d(mvis_.real, SRF_kernel, axis=0, mode='nearest') + \
             1j*convolve1d(mvis_.imag, SRF_kernel, axis=0, mode='nearest')

    # return the dataset after replacing the visibilities with the model
    if return_holders:
        # remove pads
        p_mvis = p_mvis[chpad:-chpad,:]

        # populate both polarizations
        p_mvis = np.tile(p_mvis, (2, 1, 1))

        return p_mvis, gcf, corr, dataset

    elif noise_inject is not None:
        # inject noise before SRF convolution 
        mvis_ += noise
        n_mvis = convolve1d(mvis_.real, SRF_kernel, axis=0, mode='nearest') + \
                 1j*convolve1d(mvis_.imag, SRF_kernel, axis=0, mode='nearest')

        # remove pads
        p_mvis = p_mvis[chpad:-chpad,:]
        n_mvis = n_mvis[chpad:-chpad,:]

        # populate both polarizations
        p_mvis = np.tile(p_mvis, (2, 1, 1))
        n_mvis = np.tile(n_mvis, (2, 1, 1))

        return p_mvis, n_mvis, dataset

    else:
        # remove pads
        p_mvis = p_mvis[chpad:-chpad,:]

        # populate both polarizations
        p_mvis = np.tile(p_mvis, (2, 1, 1))

        return p_mvis, dataset
