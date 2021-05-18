"""
tbd

"""

import os, sys, time
import numpy as np
from astropy.io import fits
from vis_sample.classes import *
from simple_disk import simple_disk

def cube_parser(pars, FOV=8, Npix=128, dist=150, r_min=0, r_max=500, r0=10,
                RA=240, DEC=-40, restfreq=230.538e9, Vsys=0, vel=None,
                datafile=None, outfile=None):
                
    # constants
    CC = 2.9979245800000e10
    KK = 1.38066e-16
    CC = 2.99792e10


    # generate an emission model
    disk = simple_disk(pars[0], pars[1], x0=0, y0=0, dist=dist, mstar=pars[2], 
                       r_min=r_min, r_max=r_max, r0=r0, r_l=pars[3],
                       z0=pars[4], zpsi=pars[5], zphi=np.inf, 
                       Tb0=pars[6], Tbq=pars[7], Tbeps=np.inf, Tbmax=1000, 
                       Tbmax_b=20, tau0=1000, tauq=0, taueta=np.inf, 
                       taumax=10000, dV0=None, dVq=None, 
                       dVmax=1000, xi_nt=0, FOV=FOV, Npix=Npix, mu_l=28)


    # decide on velocities
    if datafile is not None:
        # load datafile header
        dat = fits.open(datafile)
        hdr = dat[0].header

        # frequencies
        freq0 = hdr['CRVAL4']
        indx0 = hdr['CRPIX4']
        nchan = hdr['NAXIS4']
        dfreq = hdr['CDELT4']
        freqs = freq0 + (np.arange(nchan) - indx0 + 1) * dfreq

        # velocities
        vel = CC * (1. - freqs / restfreq) / 100.
    else:
        freqs = restfreq * (1. - vel / (CC / 100.))     


    # adjust for systemic velocity
    vlsr = vel - Vsys


    # generate channel maps
    cube = disk.get_cube(vlsr)


    # convert from brightness temperatures to Jy / pixel
    pixel_area = (disk.cell_sky * np.pi / (180. * 3600.))**2
    for i in range(len(freqs)):
        cube[i,:,:] *= 1e23 * pixel_area * 2 * freqs[i]**2 * KK / CC**2


    # if an 'outfile' specified, pack the cube into a FITS file 
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
