"""
An example of how to use simple_disk to make a parametric disk model.
"""

import os, sys
import numpy as np
import scipy.constants as sc
from radmc_disk import radmc_structure
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
    restfreq, FOV, npix, dist, cfg_dict = pars_fixed

    inc, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, hs_T, ws_T, Sigma0_gas, \
        p1, xmol, depl, ab_zrmin, ab_zrmax, ab_rmin, ab_rmax, xi, \
        vlsr, dx, dy = pars

    # Fixed and adjusted parameters
    r0 = 10 * _AU
    p2 = np.inf
    Tmin, Tmax = 5., 1500.
    


    # Set up the temperature structure function
    def T_gas(r, z):
        Tmid, Tatm = Tmid0 * (r / r0)**qmid, Tatm0 * (r / r0)**qatm	
        H_mid = np.sqrt(_k * Tmid / (_mu * _mH))
        hs_p, ws_p = H_mid * hs_T / r, H_mid * ws_T / r
        fz = 0.5 * np.tanh(((z / r) - hs_p) / ws_p) + 0.5
        Tout = (Tmid**4 + fz * Tatm**4)**0.25
        return np.clip(Tout, a_min=Tmin, a_max=Tmax)

    # Set up the surface density function
    def Sigma_gas(r):
        return Sigma0_gas * (r / r0)**p1 * np.exp(-(r / (r_l * _AU))**p2)

    # Set up the Keplerian angular velocity
    def omega_Kep(r, z):
        return np.sqrt(_G * (mstar * _msun) / np.hypot(r, z)**3)

    # Set up the abundance function
    def abund(r, z):
        zr_mask = np.logical_and(z/r <= ab_zrmax, z/r >= ab_zrmin)
        r_mask  = np.logical_and(r >= (ab_rmin * _AU), r <= (ab_rmax * _AU))
        return np.where(np.logical_and(zr_mask, r_mask), xmol, xmol * depl)

    # Set up the nonthermal line-width function (NEEDS WORK!)
    def nonthermal_linewidth(r, z):
        return np.zeros_like(r) 


    # Compute disk structure
    struct = radmc_structure(cfg_dict, T_gas, Sigma_gas, omega_Kep, abund,
                             nonthermal_linewidth)

    # Build the datacube
    cube = struct.get_cube(inc, PA, dist, restfreq, FOV, npix, 
                           velax=velax, vlsr=vlsr)


    # Pack the cube into a vis_sample SkyImage object and return
    mod_data = np.fliplr(np.rollaxis(cube, 0, 3))
    mod_ra  = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix) 
    mod_dec = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix)
    freq = restfreq * (1 - velax / sc.c)

    return SkyImage(mod_data, mod_ra, mod_dec, freq, None)
