import os, sys
import numpy as np
import scipy.constants as sc
from scipy.special import ellipk, ellipe


### MODEL
# Keplerian + self-gravity model (Veronesi et al.)
# r is in AU
# pars = [Mstar, Mdisk, log(f)]
def model_vphi(r, pars):

    # definitions (i.e., fixed parameters)
    dpc = 150.
    ps = np.array([0.32087389, 1.29875728, 4.34989889, 1.41132345])
    r0, rd, pp, gam = 10., 300., 1.0, np.infty

    # assign the CO emission surface (here FIXED)
    zCO = dpc * ps[0] * (r / dpc)**ps[1] * np.exp(-(r / (dpc * ps[2]))**ps[3])

    ### compute the Keplerian angular velocity
    om_kep2 = sc.G * pars[0] * 1.9891e30 / \
              np.hypot((r * sc.au), (zCO * sc.au))**3

    ### compute the non-Keplerian contribution from self-gravity
    # rprime grid
    rp = np.logspace(-1, 3, 4096)

    # k coordinates
    kk = np.sqrt(4 * rp * r[:,None] / \
                 ((r[:,None] + rp)**2 + zCO[:,None]**2))

    # xi(k)
    xik = ellipk(kk) - 0.25 * (kk**2 / (1. - kk**2)) * \
          ((rp / r[:,None]) - (r[:,None] / rp) + \
           (zCO[:,None]**2 / (r[:,None] * rp))) * ellipe(kk)

    # gas surface density on rprime grid
    sigma_ = (rp / r0)**-pp * np.exp(-(rp / rd)**gam)
    snorm_ = pars[1] * 1.9891e30 / \
             np.trapz(2 * np.pi * sigma_ * rp * sc.au, rp * sc.au)
    sigma = snorm_ * sigma_

    # field integrand
    finteg = xik * np.sqrt(rp / r[:,None]) * kk * sigma

    # field
    dPhidr = sc.G * np.trapz(finteg, rp * sc.au, axis=-1) / (r * sc.au)

    return r * sc.au * np.sqrt(om_kep2 + dPhidr / (r * sc.au))
