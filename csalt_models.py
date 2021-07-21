"""
    csalt_models.py

    Usage: 
        - import modules

    Outputs:
	- various
"""
import os, sys
import numpy as np
from astropy.io import fits
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d
from vis_sample.classes import *
from simple_disk import simple_disk
import const as const
import matplotlib.pyplot as plt


def cube_parser(pars, FOV=8, Npix=128, dist=150, r_min=0, r_max=500, r0=10,
                RA=240, DEC=-40, restfreq=230.538e9, Vsys=0, vel=None,
                datafile=None, outfile=None):

    ### Generate a model disk
    disk = simple_disk(pars[0], pars[1], x0=0, y0=0, dist=dist, mstar=pars[2], 
                       r_min=r_min, r_max=r_max, r0=r0, r_l=pars[3],
                       z0=pars[4], zpsi=pars[5], zphi=np.inf, 
                       Tb0=pars[6], Tbq=pars[7], Tbeps=np.inf, Tbmax=1000, 
                       Tbmax_b=pars[8], tau0=1000, tauq=0, taueta=np.inf, 
                       taumax=5000, dV0=pars[9], dVq=0.5*pars[7], dVmax=1000, 
                       FOV=FOV, Npix=Npix)


    ### Set velocities for cube (either use the channels in an already-existing
    ### cube from a .FITS file, or use the provided values)
    if datafile is not None:
        hd = fits.open(datafile)[0].header
        f0, ix, nf, df = hd['CRVAL4'], hd['CRPIX4'], hd['NAXIS4'], hd['CDELT4']
        freqs = f0 + (np.arange(nf) - ix + 1) * df
        vel = const.c_ * (1 - freqs / restfreq)
    else:
        freqs = restfreq * (1 - vel / const.c_)     


    # adjust for systemic velocity
    vlsr = vel - Vsys


    ### Generate the spectral line cube
    cube = disk.get_cube(vlsr)

    # convert from brightness temperatures to Jy / pixel
    pixel_area = (disk.cell_sky * np.pi / (180 * 3600))**2
    for i in range(len(freqs)):
        cube[i,:,:] *= 1e26 * pixel_area * 2 * freqs[i]**2 * \
                       const.k_ / const.c_**2


    ### Prepare the output: either into the specified .FITS file or into a 
    ### vis_sample "SKY OBJECT".
    if outfile is not None:
        hdu = fits.PrimaryHDU(cube[:,::-1,:])
        header = hdu.header
    
        # basic header inputs
        header['EPOCH'] = 2000.
        header['EQUINOX'] = 2000.
        header['LATPOLE'] = -1.436915713634E+01
        header['LONPOLE'] = 180.

        # spatial coordinates
        header['CTYPE1'] = 'RA---SIN'
        header['CUNIT1'] = 'DEG'
        header['CDELT1'] = -disk.cell_sky / 3600.
        header['CRPIX1'] = 0.5 * disk.Npix + 0.5
        header['CRVAL1'] = RA
        header['CTYPE2'] = 'DEC--SIN'
        header['CUNIT2'] = 'DEG'
        header['CDELT2'] = disk.cell_sky / 3600.
        header['CRPIX2'] = 0.5 * disk.Npix + 0.5
        header['CRVAL2'] = DEC

        # frequency coordinates
        header['CTYPE3'] = 'FREQ'
        header['CUNIT3'] = 'Hz'
        header['CRPIX3'] = 1.
        header['CDELT3'] = freqs[1]-freqs[0]
        header['CRVAL3'] = freqs[0]
        header['SPECSYS'] = 'LSRK'
        header['VELREF'] = 257

        # intensity units
        header['BSCALE'] = 1.
        header['BZERO'] = 0.
        header['BUNIT'] = 'JY/PIXEL'
        header['BTYPE'] = 'Intensity'

        # output FITS
        hdu.writeto(outfile, overwrite=True)

        return cube[:,::-1,:]

    # otherwise, return a vis_sample SkyObject
    else:
        # adjust cube formatting
        mod_data = np.rollaxis(cube[:,::-1,:], 0, 3)

        # spatial coordinates
        npix_ra = disk.Npix
        mid_pix_ra = 0.5 * disk.Npix + 0.5
        delt_ra = -disk.cell_sky / 3600
        if (delt_ra < 0):
            mod_data = np.fliplr(mod_data)
        mod_ra = (np.arange(npix_ra) - (mid_pix_ra-0.5))*np.abs(delt_ra)*3600
        
        npix_dec = disk.Npix
        mid_pix_dec = 0.5 * disk.Npix + 0.5
        delt_dec = disk.cell_sky / 3600
        if (delt_dec < 0):
            mod_data = np.flipud(mod_data)
        mod_dec = (np.arange(npix_dec)-(mid_pix_dec-0.5))*np.abs(delt_dec)*3600

        # spectral coordinates
        try:
            nchan_freq = len(freqs)
            mid_chan_freq = freqs[0]
            mid_chan = 1
            delt_freq = freqs[1] - freqs[0]
            mod_freqs = (np.arange(nchan_freq)-(mid_chan-1))*delt_freq + \
                        mid_chan_freq
        except:
            mod_freqs = [0]

        # return a vis_sample SkyImage object
        return SkyImage(mod_data, mod_ra, mod_dec, mod_freqs, None)






def vismodel_full(pars, fixed, dataset, 
                  chpad=3, oversample=None, noise_inject=None):

    ### - Prepare inputs
    # Parse fixed parameters
    restfreq, FOV, Npix, dist, rmax = fixed
    npars = len(pars)

    # Spatial frequencies to lambda units
    uu = dataset.um * np.mean(dataset.nu_TOPO) / const.c_
    vv = dataset.vm * np.mean(dataset.nu_TOPO) / const.c_

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
    v_LSRK = const.c_ * (1 - nu_LSRK / restfreq)


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
        cube = cube_parser(pars[:npars-3], FOV=FOV, Npix=Npix, dist=dist, 
                           r_max=rmax, Vsys=pars[10],
                           vel=v_LSRK[itime,:], restfreq=restfreq)

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
    chix = np.arange(nch) / oversample
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
    uu = dataset.um * np.mean(dataset.nu_TOPO) / const.c_
    vv = dataset.vm * np.mean(dataset.nu_TOPO) / const.c_

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

    # LSRK velocities at midpoint of execution block
    mid_stamp = np.int(dataset.nu_LSRK.shape[0] / 2)
    v_model = const.c_ * (1 - dataset.nu_LSRK[mid_stamp,:] / restfreq)
    v_grid = const.c_ * (1 - dataset.nu_LSRK / restfreq)

    # generate a model cube
    mcube = cube_parser(pars[:npars-3], FOV=FOV, Npix=Npix, dist=dist,
                        r_max=rmax, Vsys=pars[10], restfreq=restfreq,
                        vel=v_model)

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
