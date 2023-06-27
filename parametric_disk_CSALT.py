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

    # Parameters
    # [0] = inclination (degrees)
    # [1] = PA (degrees)
    # [2] = outer edge (au)
    # [3] = front surface height normalization (" at r = 1")
    # [4] = front surface height gradient
    # [5] = back surface height normalization (" at r = 1")
    # [6] = back surface height gradient
    # [7] = front surface temperature at 10 au (K)
    # [8] = front surface temperature gradient
    # [9] = back surface temperature at 10 au (K)
    # [10] = back surface temperature gradient
    # [11] = midplane temperature at 10 au (K)
    # [12] = midplane temperature gradient
    inc, PA, mstar, \
    r_l, zf_0, zf_q, zb_0, zb_q, \
    Tf_0, Tf_q, Tb_0, Tb_q, Tm_0, Tm_q, \
    dVnt, \
    tauf_0, tauf_q, taub_0, taub_q, taum_0, taum_q, \
    vlsr, dx, dy = pars


    # Fixed and adjusted parameters
    Tmax = 1000
    dVmax = 2 * np.sqrt(2 * sc.k * Tmax / (28 * (sc.m_p + sc.m_e)))
    x0, y0 = 0., 0.
    z_tap, T_tap, tau_tap = np.inf, r_l, r_l
    z_exp, T_exp, tau_exp = np.inf, np.inf, np.inf
    

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
        return tapered_powerlaw(r, zf_0, 1., zf_q, z_tap, z_exp)

    def z_b(r):
        """Emission surface for the back side of the disk."""
        return -tapered_powerlaw(r, zb_0, 1., zb_q, z_tap, z_exp)

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_f, side='front')

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_b, side='back')

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=None, side='mid')


    # Set up the emission profiles. For each side of the disk the emission is
    # described by an optical depth, a line width, a peak line brightness.
    def Tgas_f(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tf_0, 10., Tf_q, T_tap, T_exp)

    def Tgas_b(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tb_0, 10., Tb_q, T_tap, T_exp)

    def Tgas_m(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tm_0, 10., Tm_q, T_tap, T_exp)

    def dV_f(r):
        """Doppler linewidth profile in [m/s]."""
        return np.sqrt((2 * sc.k * Tgas_f(r) / (28 * (sc.m_p + sc.m_e))) * \
                       (1.0 + dVnt))

    def dV_b(r):
        """Doppler linewidth profile in [m/s]."""
        return np.sqrt((2 * sc.k * Tgas_b(r) / (28 * (sc.m_p + sc.m_e))) * \
                       (1.0 + dVnt))

    def dV_m(r):
        """Doppler linewidth profile in [m/s]."""
        return np.sqrt((2 * sc.k * Tgas_m(r) / (28 * (sc.m_p + sc.m_e))) * \
                       (1.0 + dVnt))

    def tau_f(r):
        """Optical depth profile."""
        return tapered_powerlaw(r, 10**tauf_0, 10., tauf_q, tau_tap, tau_exp)

    def tau_b(r):
        """Optical depth profile."""
        return tapered_powerlaw(r, 10**taub_0, 10., taub_q, tau_tap, tau_exp)

    def tau_m(r):
        """Optical depth profile."""
        tau_i = tapered_powerlaw(r, 10**taum_0, 10., taum_q, tau_tap, tau_exp)
        tau_i[Tgas_m(r) < 20] *= 1e-3
        return tau_i


    # For each of these values we can set limits which can be a useful way to
    # deal with the divergence of power laws close to the disk center. Although
    # this can be specified for each side independently, we assume they're the
    # same for simplicity (so setting `side='both'`).
    disk.set_Tgas_profile(function=Tgas_f, min=0.0, max=Tmax, side='front')
    disk.set_Tgas_profile(function=Tgas_b, min=0.0, max=Tmax, side='back')
    disk.set_Tgas_profile(function=Tgas_m, min=0.0, max=Tmax, side='mid')
    disk.set_dV_profile(function=dV_f, min=0.0, max=dVmax, side='front')
    disk.set_dV_profile(function=dV_b, min=0.0, max=dVmax, side='back')
    disk.set_dV_profile(function=dV_m, min=0.0, max=dVmax, side='mid')
    disk.set_tau_profile(function=tau_f, min=0.0, max=None, side='front')
    disk.set_tau_profile(function=tau_b, min=0.0, max=None, side='back')
    disk.set_tau_profile(function=tau_m, min=0.0, max=None, side='mid')


    # Set up the velocity structure. Here we use a simple Keplerian rotation
    # curve, although in principle anything can be used.
    def vkep(r, z):
        """Keplerian rotational velocity profile in [m/s]."""
#        r_m, z_m = r * sc.au, z_f(r / dist) * dist * sc.au
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
