from csalt.csalt_disk import csalt_disk
import scipy.constants as sc
import numpy as np
from vis_sample.classes import SkyImage
import matplotlib.pyplot as plt



def tapered_powerlaw(r, y_0, r_0, y_q, y_tap, y_exp):
    """Exponentially tapered power law."""
    return y_0 * (r / r_0)**y_q * np.exp(-(r / y_tap)**y_exp)


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
    inc, PA, mstar, r_l, z_10, z_q, Tb_10, Tb_q, Tmax_b, dV_10, \
        logtau_10, tau_q, vlsr, dx, dy = pars


    # Fixed and adjusted parameters
    dV_q = 0.5 * Tb_q
    Tmax_f = 1000
    dVmax_f = np.sqrt(2 * sc.k * Tmax_f / (28 * (sc.m_p + sc.m_e)))
    dVmax_b = np.sqrt(2 * sc.k * Tmax_b / (28 * (sc.m_p + sc.m_e)))
    x0, y0 = 0., 0.
    z_tap, Tb_tap, dV_tap, tau_tap = np.inf, r_l, r_l, r_l
    z_exp, Tb_exp, dV_exp, tau_exp = np.inf, np.inf, np.inf, np.inf
    

    # Get a simple_disk instance.
    disk = csalt_disk(quiet=quiet)


    # Set up the viewing geometry, specifying the field of view (FOV) and the
    # number of pixels on each image edge (npix). This then defines the cell
    # scale for the images.
    disk.set_sky_coords(FOV=FOV, npix=npix)


    # Set up the disk geometry. We assume the front and back side of the disk
    # are symmetric.
    def z_f(r):
        """Emission surface of the front side of the disk."""
        return tapered_powerlaw(r, z_10, 1., z_q, z_tap, z_exp)

    def z_b(r):
        """Emission surface for the back side of the disk."""
        return -z_f(r)

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_f, side='front')

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_b, side='back')


    # Set up the emission profiles. For each side of the disk the emission is
    # described by an optical depth, a line width, a peak line brightness.
    def Tgas(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tb_10, 10., Tb_q, Tb_tap, Tb_exp)

    def dV(r):
        """Doppler linewidth profile in [m/s]."""
        return tapered_powerlaw(r, dV_10, 10., dV_q, dV_tap, dV_exp)

    def tau(r):
        """Optical depth profile."""
        return tapered_powerlaw(r, 10**logtau_10, 10., tau_q, tau_tap, tau_exp)


    # For each of these values we can set limits which can be a useful way to
    # deal with the divergence of power laws close to the disk center. Although
    # this can be specified for each side independently, we assume they're the
    # same for simplicity (so setting `side='both'`).
    disk.set_Tgas_profile(function=Tgas, min=0.0, max=Tmax_f, side='front')
    disk.set_Tgas_profile(function=Tgas, min=0.0, max=Tmax_b, side='back')
    disk.set_dV_profile(function=dV, min=0.0, max=dVmax_f, side='front')
    disk.set_dV_profile(function=dV, min=0.0, max=dVmax_b, side='back')
    disk.set_tau_profile(function=tau, min=0.0, max=None, side='both')


    # Set up the velocity structure. Here we use a simple Keplerian rotation
    # curve, although in principle anything can be used.
    def vkep(r):
        """Keplerian rotational velocity profile in [m/s]."""
        r_m, z_m = r * sc.au, z_f(r / dist) * dist * sc.au
        vv = np.sqrt(sc.G * mstar * 1.98847e30 * r_m**2 / \
                     np.power(r_m**2 + z_m**2, 1.5))
        return vv

    disk.set_vtheta_profile(function=vkep, side='both')

    # Build the datacube.
    cube = disk.get_cube(velax=velax, restfreq=restfreq, vlsr=vlsr)
    cube = np.nan_to_num(cube)

    # Convert to standard surface brightness units
    freq = restfreq * (1 - velax / sc.c)

    # Pack the cube into a vis_sample SkyImage object and return
    mod_data = np.rollaxis(cube, 0, 3)
    mod_ra  = disk.cell_sky * (np.arange(npix) - 0.5 * npix) 
    mod_dec = disk.cell_sky * (np.arange(npix) - 0.5 * npix)

    return SkyImage(mod_data, mod_ra, mod_dec, freq, None)
