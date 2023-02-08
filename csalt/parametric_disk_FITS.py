"""
An example of how to use simple_disk to make a parametric disk model.
"""

import os, sys
import numpy as np
import scipy.constants as sc
from astropy.io import fits
from vis_sample.classes import SkyImage


# constants
_msun = 1.989e33
_AU = sc.au * 1e2
_mu = 2.37
_mH = (sc.m_e + sc.m_p) * 1e3
_k  = sc.k * 1e7
_G  = sc.G * 1e3


def parametric_disk(velax, pars, pars_fixed, quiet=True):
    """
    Build a parametric disk.

    Args:
        TBD

    Returns:
        TBD
    """
    # Parse the inputs
    nu_rest, FOV, npix, dist, cfg_dict = pars_fixed

    # Load the user-input FITS cube
    hdu = fits.open(pars)
    im_cube, hd = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # Get spatial (offset) coordinates from FITS cube
    dx = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
    dy = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))

    # Get spectral coordinates from FITS cube
    nu_cube = hd['CRVAL3'] + \
              hd['CDELT3'] * (np.arange(hd['NAXIS3']) - (hd['CRPIX3'] - 1))

    # Re-orient cube array
    cube = np.rollaxis(im_cube, 0, 3)

    return SkyImage(cube, dx, dy, nu_cube, None)
