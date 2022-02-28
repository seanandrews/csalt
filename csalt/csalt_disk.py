"""
Radiative transfer without the transfer.
Make quick emission maps for a disk model:

    1 - Set up the sky frame coordintes.
    2 - Set up the system geometry.
    3 - Define the emission profiles.
    4 - Produce cubes.
"""

import numpy as np
import scipy.constants as sc
import warnings


class csalt_disk:
    """
    Args:
        quiet (optional[bool]): Suppress warnings.
    """

    def __init__(self, quiet=True):
        if quiet:
            warnings.filterwarnings("ignore")

    def gaussian(self, x, x0, dx, A):
        """Simple Gaussian function."""
        return A * np.exp(-0.5*((x - x0) / dx)**2)

    # -- sky frame coordinates -- #

    def set_sky_coords(self, FOV, npix):
        """
        Populate the sky frame pixel coordinates. Distances in [arcsec].
        """
        self.x_sky = np.linspace(-FOV / 2.0, FOV / 2.0, npix)
        self.x_sky, self.y_sky = np.meshgrid(self.x_sky, self.x_sky)
        self.cell_sky = np.diff(self.x_sky).mean()
        self.npix = npix
        self.FOV = FOV

    @property
    def xaxis(self):
        return self.x_sky[0]

    @property
    def yaxis(self):
        return self.y_sky[:, 0]

    @property
    def extent(self):
        return [self.xaxis[0], self.xaxis[-1], self.yaxis[0], self.yaxis[-1]]

    # -- disk frame coordinates -- #

    def set_disk_coords(self, x0, y0, inc, PA, dist, z_func=None, side='both'):
        """
        Populate the disk frame pixel coordinates. All radial distances are in
        [au] and angles are in [rad].

        Args:
            x0 (float): RA offset of disk center in [arcsec].
            y0 (float): Dec offset of disk center in [arcsec].
            inc (float): Disk inclination in [deg].
            PA (float): Disk position angle (angle between North and the
                redshifted major axis in an easterly direction) in [deg].
            dist (float): Source distance in [pc].
            z_func (callable): A function that returns the emission surface in
                [au] for a cylindrical radius in [au].
            side (str): Which side the properties define. Must be ``'front'``,
                ``'back'`` or ``'both'``. If ``'both'``, assumes that the disk
                is 2D and both front and back coordinates are populated.
        """
        r, t, z = self._deproject(x0, y0, inc, PA, z_func)
        if side.lower() == 'both':
            self.r_disk_f, self.t_disk_f, self.z_disk_f = r * dist, t, z * dist
            self.x0_f, self.y0_f = x0, y0
            self.inc_f, self.PA_f = inc, PA
            self.dist_f = dist
            self.x_disk_f = self.r_disk_f * np.cos(self.t_disk_f)
            self.y_disk_f = self.r_disk_f * np.sin(self.t_disk_f)
            self.r_disk_b, self.t_disk_b, self.z_disk_b = r * dist, t, z * dist
            self.x0_b, self.y0_b = x0, y0
            self.inc_b, self.PA_b = inc, PA
            self.dist_b = dist
            self.x_disk_b = self.r_disk_b * np.cos(self.t_disk_b)
            self.y_disk_b = self.r_disk_b * np.sin(self.t_disk_b)
        elif side.lower() == 'front':
            self.r_disk_f, self.t_disk_f, self.z_disk_f = r * dist, t, z * dist
            self.x0_f, self.y0_f = x0, y0
            self.inc_f, self.PA_f = inc, PA
            self.dist_f = dist
            self.x_disk_f = self.r_disk_f * np.cos(self.t_disk_f)
            self.y_disk_f = self.r_disk_f * np.sin(self.t_disk_f)
        elif side.lower() == 'back':
            self.r_disk_b, self.t_disk_b, self.z_disk_b = r * dist, t, z * dist
            self.x0_b, self.y0_b = x0, y0
            self.inc_b, self.PA_b = inc, PA
            self.dist_b = dist
            self.x_disk_b = self.r_disk_b * np.cos(self.t_disk_b)
            self.y_disk_b = self.r_disk_b * np.sin(self.t_disk_b)
        else:
            raise ValueError("`side` must be 'front', 'back' or None.")

    def _deproject(self, x0, y0, inc, PA, z_func=None, frame='cylindrical'):
        """
        Deproject the sky frame coordinates into disk frame coordinates given
        geometrical parameters and an emission surface. Returns all distance
        values in [arcsec].
        """
        if frame.lower() not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")
        if z_func is None:
            r, t = self._get_midplane_polar_coords(x0, y0, inc, PA)
            z = np.zeros(r.shape)
        else:
            r, t, z = self._get_flared_coords(x0, y0, inc, PA, z_func)
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    @staticmethod
    def _get_rotated_coords(x, y, PA):
        """
        Rotate (x, y) by PA [deg].
        """
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _get_deprojected_coords(x, y, inc):
        """
        Deproject (x, y) by inc [deg].
        """
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """
        Return caresian sky coordinates in [arcsec, arcsec].
        """
        return self.x_sky - x0, self.y_sky - y0

    def _get_polar_sky_coords(self, x0, y0):
        """
        Return polar sky coordinates in [arcsec, radians].
        """
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """
        Return cartesian coordaintes of midplane in [arcsec, arcsec].
        """
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = self._get_rotated_coords(x_sky, y_sky, PA)
        return self._get_deprojected_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """
        Return the polar coordinates of midplane in [arcsec, radians].
        """
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_coords(self, x0, y0, inc, PA, z_func):
        """
        Return cylindrical coordinates of surface in [arcsec, radians].
        """
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(5):
            y_tmp = y_mid + z_func(r_tmp) * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    @property
    def x0(self):
        try:
            if self.x0_f == self.x0_b:
                return self.x0_f
            else:
                raise ValueError("x0_f != x0_b")
        except AttributeError:
            raise ValueError("Must specify both front and back surfaces.")

    @property
    def y0(self):
        try:
            if self.y0_f == self.y0_b:
                return self.y0_f
            else:
                raise ValueError("y0_f != y0_b")
        except AttributeError:
            raise ValueError("Must specify both front and back surfaces.")

    @property
    def inc(self):
        try:
            if self.inc_f == self.inc_b:
                return self.inc_f
            else:
                raise ValueError("inc_f != inc_b")
        except AttributeError:
            raise ValueError("Must specify both front and back surfaces.")

    @property
    def PA(self):
        try:
            if self.PA_f == self.PA_b:
                return self.PA_f
            else:
                raise ValueError("PA_f != PA_b")
        except AttributeError:
            raise ValueError("Must specify both front and back surfaces.")

    @property
    def dist(self):
        try:
            if self.dist_f == self.dist_b:
                return self.dist_f
            else:
                raise ValueError("dist_f != dist_b")
        except AttributeError:
            raise ValueError("Must specify both front and back surfaces.")

    # -- emission profiles -- #

    def set_Tgas_profile(self, function, side='both', min=0.0, max=None):
        """
        Populate the Doppler linewidth profiles.

        Args:
            function (callable): A function that returns the Doppler linewidth
                in [m/s] for a cylindrical radius in [au].
            side (optional[str]): The side of the disk this profile describes.
                Must be ``'both'``, ``'front'`` or ``'back'``.
            min (optional[float]): Minimum value for the linewidth in [m/s].
            max (optional[float]): Maximum value for the linewidth in [m/s].
        """
        if side.lower() == 'both':
            self.Tgas_f = np.clip(function(self.r_disk_f), a_min=min, a_max=max)
            self.Tgas_b = np.clip(function(self.r_disk_b), a_min=min, a_max=max)
        elif side.lower() == 'front':
            self.Tgas_f = np.clip(function(self.r_disk_f), a_min=min, a_max=max)
        elif side.lower() == 'back':
            self.Tgas_b = np.clip(function(self.r_disk_b), a_min=min, a_max=max)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")

    def set_dV_profile(self, function, side='both', min=0.0, max=None):
        """
        Populate the linewidth profiles.
        """
        if side.lower() == 'both':
            self.dV_f = np.clip(function(self.r_disk_f), a_min=min, a_max=max)
            self.dV_b = np.clip(function(self.r_disk_b), a_min=min, a_max=max)
        elif side.lower() == 'front':
            self.dV_f = np.clip(function(self.r_disk_f), a_min=min, a_max=max)
        elif side.lower() == 'back':
            self.dV_b = np.clip(function(self.r_disk_b), a_min=min, a_max=max)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")

    def set_tau_profile(self, function, side='both', min=0.0, max=None):
        """
        Populate the optical depth profiles.
        """
        if side.lower() == 'both':
            self.tau_f = np.clip(function(self.r_disk_f), a_min=min, a_max=max)
            self.tau_b = np.clip(function(self.r_disk_b), a_min=min, a_max=max)
        elif side.lower() == 'front':
            self.tau_f = np.clip(function(self.r_disk_f), a_min=min, a_max=max)
        elif side.lower() == 'back':
            self.tau_b = np.clip(function(self.r_disk_b), a_min=min, a_max=max)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")

    def set_vtheta_profile(self, function, side='both', min=0.0, max=None):
        """
        Populate the rotation velocity profiles.
        """
        if side.lower() == 'both':
            self.vtheta_f = np.clip(function(self.r_disk_f),
                                    a_min=min, a_max=max)
            self.vtheta_b = np.clip(function(self.r_disk_b),
                                    a_min=min, a_max=max)
        elif side.lower() == 'front':
            self.vtheta_f = np.clip(function(self.r_disk_f),
                                    a_min=min, a_max=max)
        elif side.lower() == 'back':
            self.vtheta_b = np.clip(function(self.r_disk_b),
                                    a_min=min, a_max=max)
        else:
            raise ValueError("`side` must be 'front', 'back' or 'both'.")

    # -- cube generation -- #

    @property
    def vtheta_f_proj(self):
        vt = self.vtheta_f * np.cos(self.t_disk_f)
        return vt * np.sin(abs(np.radians(self.inc)))

    @property
    def vtheta_b_proj(self):
        vt = self.vtheta_b * np.cos(self.t_disk_b)
        return vt * np.sin(abs(np.radians(self.inc)))

    def get_cube(self, velax, restfreq=230.538e9, vlsr=0.0):
        """
        Returns a data cube in units of [K].

        Args:
            velax (array): Array of the velocities of the channels in [m/s].
            vlsr (float): Systemic velocity in [m/s].
        """

        # frequencies
        nuax = restfreq * (1 - velax / sc.c)

        # spectrally-dependent optical depths (front surface)
        tau_nuf = self.gaussian(velax[:, None, None],
                                self.vtheta_f_proj[None, :, :] + vlsr,
                                self.dV_f[None, :, :] / np.sqrt(2.0),
                                self.tau_f[None, :, :])

        # Planck function (front surface)
        Bnu_f = (2 * sc.h * nuax[:, None, None]**3 / sc.c**2) / \
                (np.exp(sc.h * nuax[:, None, None] / \
                        (sc.k * self.Tgas_f[None, :, :])) - 1.0)

        # emission distribution (front surface)
        Inu_f = Bnu_f * (1.0 - np.exp(-tau_nuf)) / \
                (1.0 - np.exp(-self.tau_f[None, :, :]))
        Inu_f = np.where(np.isfinite(Inu_f), Inu_f, 0.0)

        # spectrally-dependent optical depths (back surface)
        tau_nub = self.gaussian(velax[:, None, None],
                                self.vtheta_b_proj[None, :, :] + vlsr,
                                self.dV_b[None, :, :] / np.sqrt(2.0),
                                self.tau_b[None, :, :])

        # Planck function (back surface)
        Bnu_b = (2 * sc.h * nuax[:, None, None]**3 / sc.c**2) / \
                (np.exp(sc.h * nuax[:, None, None] / \
                        (sc.k * self.Tgas_b[None, :, :])) - 1.0)

        # emission distribution (back surface)
        Inu_b = Bnu_b * (1.0 - np.exp(-tau_nub)) / \
                (1.0 - np.exp(-self.tau_b[None, :, :]))
        Inu_b = Inu_b * np.exp(-tau_nuf)
        Inu_b = np.where(np.isfinite(Inu_b), Inu_b, 0.0)

        # combined emission distribution, to proper Jy/pixel units
        Inu = Inu_f + Inu_b
        pix_area = (self.cell_sky * np.pi / 180. / 3600.)**2
        Inu *= 1e26 * pix_area

        return Inu

    # -- plotting convenience functions -- #

    def plot_dV_profile(self, side='front', return_fig=False):
        """
        Plot the linewidth profile.
        """
        if side.lower() == 'front':
            x = self.r_disk_f.flatten()
            y = self.dV_f.flatten()
        elif side.lower() == 'back':
            x = self.r_disk_b.flatten()
            y = self.dV_b.flatten()
        else:
            raise ValueError("`side` must be 'front' or 'back'.")
        label = 'Linewidth (m/s)'
        return self._plot_profile(x, y, label, return_fig)

    def plot_Tgas_profile(self, side='front', return_fig=False):
        """
        Plot the brightness profile.
        """
        if side.lower() == 'front':
            x = self.r_disk_f.flatten()
            y = self.Tgas_f.flatten()
        elif side.lower() == 'back':
            x = self.r_disk_b.flatten()
            y = self.Tgas_b.flatten()
        else:
            raise ValueError("`side` must be 'front' or 'back'.")
        label = 'Brightness (K)'
        return self._plot_profile(x, y, label, return_fig)

    def plot_tau_profile(self, side='front', return_fig=False):
        """
        Plot the optical depth profile.
        """
        if side.lower() == 'front':
            x = self.r_disk_f.flatten()
            y = self.tau_f.flatten()
        elif side.lower() == 'back':
            x = self.r_disk_b.flatten()
            y = self.tau_b.flatten()
        else:
            raise ValueError("`side` must be 'front' or 'back'.")
        label = 'Optical Depth'
        return self._plot_profile(x, y, label, return_fig)

    def plot_vtheta_profile(self, side='front', return_fig=False):
        """
        Plot the brightness profile.
        """
        if side.lower() == 'front':
            x = self.r_disk_f.flatten()
            y = self.vtheta_f.flatten()
        elif side.lower() == 'back':
            x = self.r_disk_b.flatten()
            y = self.vtheta_b.flatten()
        else:
            raise ValueError("`side` must be 'front' or 'back'.")
        label = 'Rotational Velocity (m/s)'
        return self._plot_profile(x, y, label, return_fig)

    def _plot_profile(self, x, y, label=None, return_fig=False):
        """
        Wrapper for the plotting of a radial profile.
        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        idx = np.argsort(x)
        ax.plot(x[idx], y[idx], lw=1.0, color='k')
        if label is not None:
            ax.set_ylabel(f'{label}')
        ax.set_xlabel('Radius (au)')
        ax.set_xlim(0, x.max())
        if return_fig:
            return fig
