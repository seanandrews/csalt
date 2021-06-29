"""
Wrapper program to compute and parse a simple emission model.  

This is something that we will probably ultimately combine / absorb into the
'simple_disk' setup for better organization, or perhaps maintain it but improve 
the interfacing with 'simple_disk'.  The "issue" now is that one would need to 
change both this script and 'simple_disk.py' (and 'mconfig.py') if you wanted 
to change the kind of model you're playing with, and that seems like too much.
"""

import os, sys, time
import numpy as np
from astropy.io import fits
from vis_sample.classes import *
from simple_disk import simple_disk
import const as const


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
