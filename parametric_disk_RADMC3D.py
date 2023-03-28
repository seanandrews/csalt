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


def parametric_disk(velax, pars, pars_fixed, 
                    newcube=True, struct_only=False, tausurf=False): 
    """
    Build a parametric disk.

    Args:
        TBD

    Returns:
        TBD
    """
    # Parse the inputs
    restfreq, FOV, npix, dist, cfg_dict = pars_fixed

    inc, PA, mstar, r_l, Tmid0, Tatm0, qmid, qatm, zq, deltaT, Sig0, \
        p1, p2, xmol, depl, Tfrz, Ncrit, rmax_abund, xi, vlsr, dx, dy = pars

    # Fixed and adjusted parameters
    r0 = 10 * _AU
    Tmin, Tmax = 0., 1000.


    # Set up the temperature structure function
    def T_gas(r, z):
        r, z = np.atleast_1d(r), np.atleast_1d(z)
        Tmid, Tatm = Tmid0 * (r / r0)**qmid, Tatm0 * (r / r0)**qatm	
        zqr = zq * r
        fz = (np.cos(np.pi * z / (2 * zqr)))**deltaT
        Tgas = Tatm + (Tmid - Tatm) * fz
        Tout = np.where(z >= zqr, Tatm, Tgas)
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
        # sophisticated boundary mode
        zcrit = np.zeros_like(r)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                # define an artificial z grid
                zg = np.linspace(0, 5.*r[i,j], 1024)

                # vertical sound speed profile
                cs = np.sqrt(_k * T_gas(r[i,j], zg) / (_mu * _mH))

                # vertical log(sound speed) gradient
                dlnc, dz = np.diff(np.log(cs)), np.diff(zg)
                dlncdz = np.append(dlnc, dlnc[-1]) / np.append(dz, dz[-1])

                # vertical gravity from star
                gz_star = omega_Kep(r[i,j], zg)**2 * zg
 
                # vertical gravity from disk
                if cfg_dict['dens_selfgrav']:
                    gz_disk = 2 * np.pi * _G * Sigma_gas(r[i,j])
                else:
                    gz_disk = 0.

                # total vertical gravity
                gz = gz_star + gz_disk

                # vertical log(density) gradient profile
                dlnpdz = -gz / cs**2 - 2 * dlncdz

                # numerical integration
                lnp = integrate.cumtrapz(dlnpdz, zg, initial=0)
                rho0 = np.exp(lnp)

                # normalize
                rho = 0.5 * rho0 * Sigma_gas(r[i,j])
                rho /= integrate.trapz(rho0, zg)

                # to number densities
                n_ = rho / (_mu * _mH)
                ngas = np.clip(n_, a_min=100, a_max=1e50)

                # flip to integrate downwards in z
                fngas, fz = ngas[::-1], zg[::-1]

                # vertically-integrated column profile
                Nz = -integrate.cumtrapz(fngas, fz, initial=0)

                # interpolate to find critical height
                zint = interp1d(Nz, fz, kind='linear', fill_value='extrapolate')
                zcrit[i,j] = zint(Ncrit)

        z_mask = np.logical_and(z <= zcrit, T_gas(r, z) >= Tfrz)
        layer_mask = np.logical_and(z_mask, (r <= rmax_abund * _AU))
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

    if struct_only:
        return 0

    if tausurf:
        print('\n Computing emission line photosphere locations... \n')
        tau_locs = struct.get_tausurf(inc, PA, dist, restfreq, FOV, npix, 
                                      taus=2./3., velax=velax, vlsr=vlsr)
        return 0

    else:
        # Build the datacube
        cube = struct.get_cube(inc, PA, dist, restfreq, FOV, npix, 
                               velax=velax, vlsr=vlsr, newcube=newcube)


        # Pack the cube into a vis_sample SkyImage object and return
        mod_data = np.rollaxis(cube, 0, 3)
        mod_ra  = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix) 
        mod_dec = (FOV / (npix - 1)) * (np.arange(npix) - 0.5 * npix)
        freq = restfreq * (1 - velax / sc.c)

        tf = time.time()
        print('radmc time = %f' % ((tf - t1) / 60.))

        return SkyImage(mod_data, mod_ra, mod_dec, freq, None)
