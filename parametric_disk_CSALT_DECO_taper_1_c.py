from csalt.csalt_DECO import csalt_disk
import scipy.constants as sc
import numpy as np
from vis_sample.classes import SkyImage
import matplotlib.pyplot as plt

#2 surfaces, taper on z and tau, with same cutoff radius


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
    inc, PA, mstar, r_l, r_l_b, z_f10, z_fq, z_b10, z_bq, \
    Tf_10, Tf_q, Tb_10, Tb_q, \
    z_exp, tau_exp, z_exp_b, tau_exp_b,  \
    logtau_10, tau_q, vlsr, dx, dy = pars


    # Fixed and adjusted parameters
    dVnt = 0
    Tmax = 1000
    dVmax = np.sqrt(2 * sc.k * Tmax / (28 * (sc.m_p + sc.m_e)))
    x0, y0 = 0., 0.
    z_tap, Tb_tap, dV_tap, tau_tap = r_l, np.inf, np.inf, r_l
    z_exp, Tb_exp, dV_exp, tau_exp = z_exp, np.inf, np.inf, tau_exp
    z_tap_b, tau_tap_b = r_l_b, r_l_b
    z_exp_b, tau_exp_b = z_exp_b, tau_exp_b

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
        return tapered_powerlaw(r, z_f10, 1., z_fq, z_tap, z_exp)

    def z_b(r):
        """Emission surface for the back side of the disk."""
        return -tapered_powerlaw(r, z_b10, 1., z_bq, z_tap_b, z_exp_b)

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_f, side='front')

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_b, side='back')


    # Set up the emission profiles. For each side of the disk the emission is
    # described by an optical depth, a line width, a peak line brightness.
    def Tgas_f(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tf_10, 10., Tf_q, Tb_tap, Tb_exp)
    
    def Tgas_b(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tb_10, 10., Tb_q, Tb_tap, Tb_exp)

    def dV_f(r):
        """Doppler linewidth profile in [m/s]."""
        return np.sqrt((2 * sc.k * Tgas_f(r) / (28 * (sc.m_p + sc.m_e))) * \
                       (1.0 + dVnt))

    def dV_b(r):
        """Doppler linewidth profile in [m/s]."""
        return np.sqrt((2 * sc.k * Tgas_b(r) / (28 * (sc.m_p + sc.m_e))) * \
                       (1.0 + dVnt))

    def tau_f(r):
        """Optical depth profile."""
        return tapered_powerlaw(r, 10**logtau_10, 10., tau_q, tau_tap, tau_exp)

    def tau_b(r):
        """Optical depth profile."""
        return tapered_powerlaw(r, 10**logtau_10, 10., tau_q, tau_tap_b, tau_exp_b)
    # For each of these values we can set limits which can be a useful way to
    # deal with the divergence of power laws close to the disk center. Although
    # this can be specified for each side independently, we assume they're the
    # same for simplicity (so setting `side='both'`).
    disk.set_Tgas_profile(function=Tgas_f, min=0.0, max=Tmax, side='front')
    disk.set_Tgas_profile(function=Tgas_b, min=0.0, max=Tmax, side='back')
    disk.set_dV_profile(function=dV_f, min=0.0, max=dVmax, side='front')
    disk.set_dV_profile(function=dV_b, min=0.0, max=dVmax, side='back')
    disk.set_tau_profile(function=tau_f, min=0.0, max=None, side='front')
    disk.set_tau_profile(function=tau_b, min=0.0, max=None, side='back')


    # Set up the velocity structure. Here we use a simple Keplerian rotation
    # curve, although in principle anything can be used.
    def vkep(r, z):
        """Keplerian rotational velocity profile in [m/s]."""
        r_m, z_m = r * sc.au, z * sc.au
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
