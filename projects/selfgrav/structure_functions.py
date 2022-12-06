import os, sys
import numpy as np
import scipy.constants as sc
from scipy import integrate
from scipy.special import ellipk, ellipe
from scipy.interpolate import interp1d

'''
STRUCTURE FUNCTIONS 

All functions assume cylindrical coordinate system, with gridpoints defined in 
SI units (meters, kilograms, seconds), and an input dictionary from a CSALT 
config file for a RADMC-3D model.
'''

# Keplerian angular velocities
def omega_kep(r, z, inp):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # return
    return np.sqrt(sc.G * inp.mstar * 1.9891e30 / np.hypot(r, z)**3)


# temperatures
def temperature(r, z, inp, r0=10*sc.au, mu=2.37, iso=False):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # midplane, atmosphere radial profiles
    Tmid = inp.Tmid0 * (r / r0)**inp.qmid
    if iso:
        Tatm = 1. * Tmid
    else:
        Tatm = inp.Tatm0 * (r / r0)**inp.qatm

    # vertical transition
    fz = 0.5 * np.tanh(((z/r) - inp.a_z) / inp.w_z) + 0.5

    # 2-D structure
    Tout = Tmid + fz * (Tatm - Tmid)

    # return clipped distribution
    return np.clip(Tout, a_min=0., a_max=1000)


# sound speeds
def c_sound(r, z, inp, r0=10*sc.au, mu=2.37):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # return
    c_s = np.sqrt(sc.k * temperature(r, z, inp, r0=r0, mu=mu) / \
                  (mu * (sc.m_e + sc.m_p)))
    return c_s


# pressure scale heights
def H_pressure(r, inp, r0=10*sc.au, mu=2.37):
    # grid maintenance
    r = np.atleast_1d(r)

    cs = c_sound(r, np.zeros_like(r), inp, r0=r0, mu=mu)
    om = omega_kep(r, np.zeros_like(r), inp)

    # return
    return cs / om


# surface densities
def sigmagas(r, inp, r0=10*sc.au):
    # note: pre-factor of 10 converts from inputs in g/cm**3 to kg/m**3
    sig = 10 * inp.Sig0 * (r / r0)**inp.p1 * \
          np.exp(-(r / (sc.au * inp.r_l))**inp.p2)

    # return clipped distribution
    return np.clip(sig, a_min=1e-50, a_max=1e50)


# number densities (all gas)
def numberdensity(r, z, inp, r0=10*sc.au, mu=2.37, n_bg=100, selfgrav=False):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # sound speeds
    cs = c_sound(r, z, inp, r0=r0, mu=mu)

    # vertical log(sound speed) gradient
    dlnc, dz = np.diff(np.log(cs), axis=0), np.diff(z, axis=0)
    dlncdz = np.row_stack((dlnc, dlnc[-1, :])) / np.row_stack((dz, dz[-1, :]))

    # vertical gravity from star
    gz_star = omega_kep(r, z, inp)**2 * z

    # vertical gravity from disk
    if selfgrav:
        gz_disk = 2 * np.pi * sc.G * sigmagas(r, inp, r0=r0)
    else:
        gz_disk = 0.

    # total gravity
    gz = gz_star + gz_disk

    # vertical log(density) gradient
    dlnpdz = -gz / cs**2 - 2 * dlncdz

    # numerical integration to log(density) [un-normalized]
    lnp = integrate.cumtrapz(dlnpdz, z, initial=0, axis=0)
    rho0 = np.exp(lnp)

    # normalize
    rho = 0.5 * rho0 * sigmagas(r, inp, r0=r0)
    rho /= integrate.trapz(rho0, z, axis=0)

    # the number density (per cm**3)
    n_ = 1e-6 * rho / (mu * (sc.m_e + sc.m_p))

    # return the clipped number density
    return np.clip(n_, a_min=n_bg, a_max=1e50)


def zcrit_pd(r, z, inp, r0=10*sc.au, mu=2.37, n_bg=100, 
             selfgrav=False):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # compute number densities (in cm**-3)
    ngas = numberdensity(r, z, inp, r0=r0, mu=mu, n_bg=n_bg, selfgrav=selfgrav)

    # find the surface
    zcrit = np.empty(r.shape[-1])
    for ir in range(r.shape[-1]):
        fngas, fz = ngas[::-1,ir], z[::-1,ir]
        Nz = -integrate.cumtrapz(fngas, 1e2 * fz, initial=0)
        zint = interp1d(Nz, fz, kind='linear', fill_value='extrapolate')
        zcrit[ir] = zint(inp.Ncrit)

    return zcrit 


def abundance(r, z, inp, r0=10*sc.au, mu=2.37, n_bg=100, selfgrav=False):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # find upper boundary
    z_hi = zcrit_pd(r, z, inp, r0=r0, mu=mu, n_bg=n_bg, selfgrav=selfgrav)

    # criteria for abundant layer
    Tgas = temperature(r, z, inp, r0=r0, mu=mu)
    z_mask = np.logical_and(z <= z_hi, Tgas >= inp.Tfrz)

    # outer boundary criterion (for non-tapered cases)
    r_mask = (r <= inp.rmax_abund*sc.au)

    # combined layer mask
    layer_mask = np.logical_and(z_mask, r_mask)
    
    return np.where(layer_mask, inp.xmol, inp.xmol * inp.depl)


def numberdensity_iso(r, z, inp, r0=10*sc.au, mu=2.37):

    # pressure scale heights
    H_P = H_pressure(r, inp, r0=r0, mu=mu)

    # density
    rho = sigmagas(r, inp, r0=r0) * np.exp(-0.5 * (z / H_P)**2) / \
          (np.sqrt(2 * np.pi) * H_P)

    # return the number density (per cm**3)
    return 1e-6 * rho / (mu * (sc.m_e + sc.m_p))


# CO abundance function
def abund(r, z, inp, r0=10*sc.au, mu=2.37):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # logic masks
    zmask = np.logical_and(z/r <= inp.zrmax, 
                           temperature(r, z, inp, r0=r0, mu=mu) >= inp.Tfrz)
    rmask = np.logical_and(r >= inp.rmin * sc.au, r <= inp.rmax * sc.au)

    # return
    return np.where(np.logical_and(zmask, rmask), inp.xmol, inp.xmol*inp.depl)


def eps_P(r, z, inp, r0=10*sc.au, mu=2.37, niso=False, nselfgrav=False):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # analytic, vertically isothermal case
    if niso:
        H_P = H_pressure(r, inp, r0=r0, mu=mu)
        qq, pp = -1 * inp.qmid, -1 * inp.p1
        epsP = (c_sound(r, np.zeros_like(z), inp, r0=r0, mu=mu)**2 / r**2) * \
               (0.5 * (3 - qq) * ((z / H_P)**2 - 1.) - (pp + qq))

    # numerical case
    else:
        if nselfgrav:
            nn = 1e6 * numberdensity(r, z, inp, r0=r0, mu=mu, selfgrav=True)
        else:
            nn = 1e6 * numberdensity(r, z, inp, r0=r0, mu=mu)
        logrho = np.log(nn * mu * (sc.m_e + sc.m_p))
        dlogrhodr = np.gradient(logrho, r[0,:], axis=1)
        cs = c_sound(r, z, inp, r0=r0, mu=mu)
        dcdr = np.gradient(cs, r[0,:], axis=1)
        epsP = cs**2 * dlogrhodr / r + 2 * cs * dcdr / r

    return epsP


def eps_g(r, z, inp, r0=10*sc.au, mu=2.37):
    # grid maintenance
    r, z = np.atleast_1d(r), np.atleast_1d(z)

    # 1-D case
    if r.ndim == 1:
        # integral grid
        rp = np.logspace(np.log10(r[0]), np.log10(r[-1]), 4096)

        # k coordinate
        kk = np.sqrt(4 * rp * r[:, None] / \
                     ((r[:,None] + rp)**2 + z[:,None]**2))

        # xi(k)
        xik = ellipk(kk) - 0.25 * (kk**2 / (1. - kk**2)) * \
              ((rp / r[:, None]) - (r[:, None] / rp) + \
               (z[:,None]**2 / (r[:,None] * rp))) * ellipe(kk)

        # field integrand
        finteg = xik * np.sqrt(rp / r[:,None]) * kk * sigmagas(rp, inp, r0=r0)

        # field
        dPhidr = sc.G * np.trapz(finteg, rp, axis=-1) / r
    
    # 2-D case
    else: 
        # integral grid
        rp = np.logspace(np.log10(r[0, 0]), np.log10(r[0, -1]), 4096)

        # k coordinate
        kk = np.sqrt(4 * rp[None, :] * r[:, :, None] / \
                     ((r[:,:,None] + rp[None,:])**2 + z[:,:,None]**2))

        # g(k)
        gk = ellipk(kk) - 0.25 * (kk**2 / (1. - kk**2)) * \
             ((rp[None, :] / r[:, :, None]) - (r[:, :, None] / rp[None, :]) + \
              (z[:,:,None]**2 / (r[:,:,None] * rp[None,:]))) * ellipe(kk)

        # field integrand
        finteg = gk * np.sqrt(rp[None,:] / r[:,:,None]) * kk * \
                 sigmagas(rp[None,:], inp, r0=r0)

        # field
        dPhidr = sc.G * np.trapz(finteg, rp, axis=-1) / r

    # return
    return dPhidr / r
