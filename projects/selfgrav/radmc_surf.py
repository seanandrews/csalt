import os, sys
import numpy as np
import scipy.constants as sc
from astropy.io import fits
from scipy.optimize import curve_fit


def radmc_surf(sdir, dpc=150, rfit_max=280, return_points=True):

    # load tau surface 3d locations
    tau_locs = np.load(sdir+'/tausurface_3d.npz')['tau_locs']

    # load raw cube
    hdu = fits.open(sdir+'/raw_cube.fits')
    Ico = np.squeeze(hdu[0].data)
    hdu.close()

    # find channels with peak intensity for each pixel
    ix_peakI = np.argmax(Ico, axis=0)

    # extract 3d surface locations at those channels for each pixel
    taux, tauy, tauz = tau_locs[0,:,:,:], tau_locs[1,:,:,:], tau_locs[2,:,:,:]
    xsurf, ysurf = np.empty(ix_peakI.shape), np.empty(ix_peakI.shape)
    zsurf = np.empty(ix_peakI.shape)
    for i in range(ix_peakI.shape[0]):
        for j in range(ix_peakI.shape[1]):
            xsurf[i, j] = taux[ix_peakI[i, j], i, j]
            ysurf[i, j] = tauy[ix_peakI[i, j], i, j]
            zsurf[i, j] = tauz[ix_peakI[i, j], i, j]
    rsurf = np.sqrt(xsurf**2 + ysurf**2)

    # surface fit conditions
    cond0 = np.logical_and(zsurf > 0, ix_peakI > 0)
    cond1 = np.logical_and(cond0, rsurf <= rfit_max * sc.au)
    condf = np.logical_and(cond1,
                       ~np.logical_and(rsurf > 200*sc.au, zsurf/rsurf <= 0.27))

    # impose the conditions
    rf, zf = (rsurf[condf] / sc.au) / dpc, (zsurf[condf] / sc.au) / dpc
    dz = 1. / (10 * np.ones_like(zf))

    # fittable, cleaned up data
    nan_mask = np.isfinite(rf) & np.isfinite(zf) & np.isfinite(dz)
    r, z, dz = rf[nan_mask], zf[nan_mask], dz[nan_mask]
    idx = np.argsort(r)
    r, z, dz = r[idx], z[idx], dz[idx]
    dist = 1.
    r0 = 1.

    # fit setup and optimization
    def _powerlaw(r, z0, q, r_cavity=0.0, r0=1.0):
        """Standard power law profile."""
        rr = np.clip(r, a_min=0.0, a_max=None)
        return z0 * (rr / r0)**q

    def _tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r0=1.0):
        """Exponentially tapered power law profile."""
        rr = np.clip(r, a_min=0.0, a_max=None)
        f = _powerlaw(rr, z0, q, r0=r0)
        return f * np.exp(-(rr / r_taper)**q_taper)

    kw = {}
    kw['maxfev'] = kw.pop('maxfev', 100000)
    kw['sigma'] = dz

    kw['p0'] = [0.3 * dist, 1.0, 1.0 * dist, 1.0]
    def func(r, *args):
        return _tapered_powerlaw(r, *args, r0=r0)

    try:
        popt, copt = curve_fit(func, r, z, **kw)
        copt = np.diag(copt)**0.5
    except RuntimeError:
        popt = kw['p0']
        copt = [np.nan for _ in popt]


    if return_points:
        return popt, rsurf, zsurf
    else:
        return popt
