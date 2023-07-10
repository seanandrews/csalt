"""
An example of how to use simple_disk to make a parametric disk model.
"""

import os, sys, time
import numpy as np
import scipy.constants as sc
from csalt.radmc_disk import radmc_structure
from vis_sample.classes import SkyImage
from scipy import integrate
from scipy.interpolate import interp1d


# constants
_msun = 1.9891e33
_AU = sc.au * 1e2
_mu = 2.37
_mH = (sc.m_e + sc.m_p) * 1e3
_k  = sc.k * 1e7
_G  = sc.G * 1e3


def parametric_disk(velax, pars, pars_fixed):
    """
    Build a parametric disk.

    Args:
        TBD

    Returns:
        TBD
    """
    # Parse the inputs
    restfreq, FOV, npix, dist, cfg_dict = pars_fixed
    if 'newcube' not in cfg_dict:
        cfg_dict['newcube'] = True
    if 'struct_only' not in cfg_dict:
        cfg_dict['struct_only'] = False
    if 'tausurf' not in cfg_dict:
        cfg_dict['tausurf'] = False

    inc, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, zq, deltaT, Sig0, \
        p1, p2, xmol, depl, ab_zlo, ab_zhi, rmax_ab, xi, vlsr, dx, dy = pars

    # Fixed and adjusted parameters
    r0 = 10 * _AU
    Tmin, Tmax = 0., 1000.

    # Set up the Keplerian angular velocity
    def omega_Kep(r, z):
        return np.sqrt(_G * (mstar * _msun) / np.hypot(r, z)**3)

    # Set up the temperature structure function
    def T_gas(r, z):
        r, z = np.atleast_1d(r), np.atleast_1d(z)
        Tmid, Tatm = Tmid0 * (r / r0)**qmid, Tatm0 * (r / r0)**qatm	
        zqr = zq * np.sqrt(_k * Tmid / (_mu * _mH)) / \
              omega_Kep(r, np.zeros_like(r))
        fz = (np.cos(np.pi * z / (2 * zqr)))**deltaT
        Tgas = Tatm + (Tmid - Tatm) * fz
        Tout = np.where(z >= zqr, Tatm, Tgas)
        return np.clip(Tout, a_min=Tmin, a_max=Tmax)

    # Set up the surface density function
    def Sigma_gas(r):
        sig = Sig0 * (r / r0)**p1 * np.exp(-(r / (r_l * _AU))**p2)
        return np.clip(sig, a_min=1e-50, a_max=1e50)

    # Set up the abundance function
    def abund(r, z):
        H = np.sqrt(_k * Tmid0 * (r / r0)**qmid / (_mu * _mH)) / \
            omega_Kep(r, np.zeros_like(r))
        z_mask = np.logical_and(z >= ab_zlo * H, z <= ab_zhi * H)
        layer_mask = np.logical_and(z_mask, (r <= rmax_ab * _AU))
        return np.where(layer_mask, xmol, xmol * depl)


    # Set up the nonthermal line-width function (NEEDS WORK!)
    def nonthermal_linewidth(r, z):
        return np.zeros_like(r) 


    # Compute and quote the total gas mass
    r_test = np.logspace(-1, np.log10(1000), 2048) * _AU
    M_gas = np.trapz(2 * np.pi * r_test * Sigma_gas(r_test), r_test) / _msun
    print('Gas mass of disk = %.4f Msun' % M_gas)

    # Compute disk structure
    t0 = time.time()
    struct = radmc_structure(cfg_dict, T_gas, Sigma_gas, omega_Kep, abund,
                             nonthermal_linewidth)
    t1 = time.time()
    print('structure time = %f' % ((t1 - t0) / 60.))

    if cfg_dict['struct_only']:
        return 0

    if cfg_dict['tausurf']:
        print('\n Computing emission line photosphere locations... \n')
        tau_locs = struct.get_tausurf(inc, PA, dist, restfreq, FOV, npix, 
                                      taus=2./3., velax=velax, vlsr=vlsr)
        return 0

    else:
        # Build the datacube
        cube = struct.get_cube(inc, PA, dist, restfreq, FOV, npix, 
                               velax=velax, vlsr=vlsr, 
                               newcube=cfg_dict['newcube'])


        # Pack the cube into a vis_sample SkyImage object and return
        mod_data = np.rollaxis(cube, 0, 3)
        mod_ra  = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix) 
        mod_dec = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix)
        freq = restfreq * (1 - velax / sc.c)

        tf = time.time()
        print('radmc time = %f' % ((tf - t1) / 60.))

        return SkyImage(mod_data, mod_ra, mod_dec, freq, None)
