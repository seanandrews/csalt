"""
An example of how to use simple_disk to make a parametric disk model.
"""

import os, sys
import numpy as np
import scipy.constants as sc
from csalt.radmc_disk import radmc_structure
from vis_sample.classes import SkyImage


# constants
_msun = 1.9891e33
_AU = sc.au * 1e2
_mu = 2.37
_mH = (sc.m_e + sc.m_p) * 1e3
_k  = sc.k * 1e7
_G  = sc.G * 1e3


def parametric_disk(velax, pars, pars_fixed, struct_only=False, quiet=True):
    """
    Build a parametric disk.

    Args:
        TBD

    Returns:
        TBD
    """
    # Parse the inputs
    restfreq, FOV, npix, dist, cfg_dict = pars_fixed

    inc, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, a_z, w_z, Sig0, \
        p1, p2, xmol, depl, Tfrz, ab_zrmax, ab_rmin, ab_rmax, xi, \
        vlsr, dx, dy = pars

    # Fixed and adjusted parameters
    r0 = 10 * _AU
    Tmin, Tmax = 0., 1000.


    # Set up the temperature structure function
    def T_gas(r, z):
        r, z = np.atleast_1d(r), np.atleast_1d(z)
        Tmid, Tatm = Tmid0 * (r / r0)**qmid, Tatm0 * (r / r0)**qatm	
        H_p = np.sqrt(_k * Tmid / (_mu * _mH)) / omega_Kep(r, np.zeros_like(r))
        if cfg_dict['selfgrav']:
            Q = np.sqrt(_k * Tmid / (_mu * _mH)) * \
                omega_Kep(r, np.zeros_like(r)) / (np.pi * _G * Sigma_gas(r))
            H = np.sqrt(np.pi / 2) * (np.pi / (4 * Q)) * \
                (np.sqrt(1 + 8 * Q**2 / np.pi) - 1) * H_p
        else:
            H = H_p
        fz = 0.5 * np.tanh(((z / r) - a_z * (H / r)) / (w_z * (H / r))) + 0.5
        Tout = Tmid + fz * (Tatm - Tmid)
        return np.clip(Tout, a_min=Tmin, a_max=Tmax)

    # Set up the surface density function
    def Sigma_gas(r):
        sig = Sig0 * (r / r0)**p1 * np.exp(-(r / (r_l * _AU))**p2)
        return np.clip(sig, a_min=1e-50, a_max=1e50)

    # Set up the Keplerian angular velocity
    def omega_Kep(r, z):
        return np.sqrt(_G * (mstar * _msun) / np.hypot(r, z)**3)

    # Set up the abundance function
    def abund(r, z):
        H_p = np.sqrt(_k * Tmid0 * (r / r0)**qmid / (_mu * _mH)) / \
              omega_Kep(r, np.zeros_like(r))
        z_mask = np.logical_and(z <= ab_zrmax * H_p, T_gas(r, z) >= Tfrz)
        #z_mask = np.logical_and(z <= ab_zrmax * H_p, z >= 0.7 * ab_zrmax * H_p)
        r_mask = np.logical_and(r >= (ab_rmin * _AU), r <= (ab_rmax * _AU))
        return np.where(np.logical_and(z_mask, r_mask), xmol, xmol * depl)

    # Set up the nonthermal line-width function (NEEDS WORK!)
    def nonthermal_linewidth(r, z):
        return np.zeros_like(r) 


    # Compute and quote the total gas mass
    r_test = np.logspace(-1, np.log10(1.1 * r_l), 2048) * _AU
    M_gas = np.trapz(2 * np.pi * r_test * Sigma_gas(r_test), r_test) / _msun
    M_analytic = 2 * np.pi * Sig0 * r0**-p1 * (r_l * _AU)**(2 + p1) / (2 + p1)
    print('Gas mass of disk = %.4f Msun' % M_gas)
    print('Analytic mass of disk = %.4f Msun' % (M_analytic / _msun))

    # Compute disk structure
    struct = radmc_structure(cfg_dict, T_gas, Sigma_gas, omega_Kep, abund,
                             nonthermal_linewidth)

    if struct_only:
        return 0

    else:
        # Build the datacube
        cube = struct.get_cube(inc, PA, dist, restfreq, FOV, npix, 
                               velax=velax, vlsr=vlsr)


        # Pack the cube into a vis_sample SkyImage object and return
        mod_data = np.rollaxis(cube, 0, 3)
        mod_ra  = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix) 
        mod_dec = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix)
        freq = restfreq * (1 - velax / sc.c)

        return SkyImage(mod_data, mod_ra, mod_dec, freq, None)
