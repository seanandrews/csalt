import numpy as np
import scipy.constants as sc
from scipy.special import erf
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel


class simple_disk:
    """
    Args:
        # Geometric Parameters
        inc (float): Inclination of the source in [degrees].
        PA (float): Position angle of the source in [degrees].
        x0 (Optional[float]): Source center offset along x-axis in [arcsec].
        y0 (Optional[float]): Source center offset along y-axis in [arcsec].
        dist (Optional[float]): Distance to the source in [pc].
        mstar (Optional[float]): Mass of the central star in [Msun].
        r_min (Optional[float]): Inner radius in [au].
        r_max (Optional[float]): Outer radius in [au].
        r0 (Optional[float]): Normalization radius in [au]. (r0 must be < r_l)
        r_l (Optional[float]): Turn-over radius in [au].
        z0 (Optional[float]): Emission height in [au] at r0.
        zpsi (Optional[float]): Index of z_l profile for r < r_l.
        zphi (Optional[float]): Exponential taper index of z_l profile at 
            r > r_l.

        # Brightness Temperatures
        Tb0 (Optional[float]): Brightness temperature in [K] at r0.
        Tbq (Optional[float]): Index of Tb profile for r < r_l.
        Tbeps (Optional[float]): Exponential taper index of Tb profile for 
            r > r_l.
        Tbmax (Optional[float]): Maximum Tb in [K].
        Tbmax_b (Optional[float]): Maximum Tb for back side of disk in [K].

        # Optical depth of front-side
        tau0 (Optional[float]): Optical depth at r0.
        tauq (Optional[float]): Index of optical depth profile for r < r_l
        taueta (Optional[float]): Exponential taper index for optical depth
            profile at r > r_l.
        taumax (Optional[float]): Maximum optical depth.

        # Line-widths
        dV0 (Optional[float]): Doppler line width in [m/s] at r0.
        dVq (Optional[float]): Index of line-width profile.
        dVmax (Optional[float]): Maximum line-width.
        xi_nt (Optional[float]): Non-thermal line-width fraction (of sound 
	    speed for the gas); can use if dV0, dVq are None.

        # Observational Parameters
        FOV (Optional[float]): Field of view of the model in [arcsec].
        Npix (Optional[int]): Number of pixels along each axis.
        mu_l (Optional[float]): Mean atomic weight for line of interest.

    """

    # Establish constants
    mu = 2.37
    msun = 1.98847e30
    mH = sc.m_p + sc.m_e

    # Establish useful conversion factors
    fwhm = 2.*np.sqrt(2.*np.log(2.))
    nwrap = 3


    def __init__(self, inc, PA, x0=0., y0=0., dist=100., mstar=1., 
                 r_min=0., r_max=1000., r0=10., r_l=100., 
                 z0=0., zpsi=1., zphi=np.inf,
                 Tb0=50., Tbq=0.5, Tbeps=np.inf, Tbmax=1000., Tbmax_b=20., 
                 tau0=100., tauq=0., taueta=np.inf, taumax=None,
                 dV0=None, dVq=None, dVmax=1000., xi_nt=0.,
                 FOV=None, Npix=128, mu_l=28):



        # Set the disk geometrical properties.
        self.x0, self.y0, self.inc, self.PA, self.dist = x0, y0, inc, PA, dist
        self.z0, self.zpsi, self.zphi = z0, zpsi, zphi
        self.r_l, self.r0, self.r_min, self.r_max = r_l, r0, r_min, r_max

        # Define the velocity, brightness and linewidth radial profiles.
        self.mstar = mstar
        self.Tb0, self.Tbq, self.Tbeps = Tb0, Tbq, Tbeps
        self.Tbmax, self.Tbmax_b = Tbmax, Tbmax_b
        self.dV0, self.dVq, self.dVmax, self.xi_nt = dV0, dVq, dVmax, xi_nt
        self.tau0, self.tauq, self.taueta = tau0, tauq, taueta
        self.taumax = taumax

        # Set the observational parameters.
        self.FOV = 2.2 * self.r_max / self.dist if FOV is None else FOV
        self.Npix = Npix
        self.mu_l = mu_l


        # Check if dV should be set by thermal broadening.
        #self._check_thermal_broadening()
        self._check_optical_depth()

        # Build the disk model.
        self._populate_coordinates()
        self._set_brightness()
        self._set_linewidth()
        self._set_rotation()
        self._set_tau()


    # -- Model Building Functions -- #

    def _populate_coordinates(self):
        """
        Populate the coordinates needed for the model.
        """

        # Set sky cartesian coordinates, representing the pixels in the image.

        self.x_sky = np.linspace(-self.FOV / 2.0, self.FOV / 2.0, self.Npix)
        self.cell_sky = np.diff(self.x_sky).mean()
        self.x_sky, self.y_sky = np.meshgrid(self.x_sky, self.x_sky)

        # Use these pixels to define face-down disk-centric coordinates.

        self.x_disk = self.x_sky * self.dist
        self.y_disk = self.y_sky * self.dist
        self.cell_disk = np.diff(self.x_disk).mean()

        # Define three sets of cylindrical coordintes, the two emission
        # surfaces and the midplane. If `z0 = 0.0` then the two emission
        # surfaces are equal.

        self.r_disk = np.hypot(self.y_disk, self.x_disk)
        self.t_disk = np.arctan2(self.y_disk, self.x_disk)

        f = self.disk_coords(x0=self.x0, y0=self.y0, inc=self.inc, PA=self.PA,
                             z0=self.z0, zpsi=self.zpsi, zphi=self.zphi)

        self.r_sky_f = f[0] * self.dist
        self.t_sky_f = f[1]
        self.z_sky_f = f[2] * self.dist

        if self.z0 != 0.0:
            self._flat_disk = False
            b = self.disk_coords(x0=self.x0, y0=self.y0, inc=-self.inc,
                                 PA=self.PA, z0=self.z0, zpsi=self.zpsi,
                                 zphi=self.zphi)
        else:
            self._flat_disk = True
            b = f

        self.r_sky_b = b[0] * self.dist
        self.t_sky_b = b[1]
        self.z_sky_b = b[2] * self.dist

        # Define masks noting where the disk extends to.

        self._in_disk_f = np.logical_and(self.r_sky_f >= self.r_min,
                                         self.r_sky_f <= self.r_max)
        self._in_disk_b = np.logical_and(self.r_sky_b >= self.r_min,
                                         self.r_sky_b <= self.r_max)
        self._in_disk = np.logical_and(self.r_disk >= self.r_min,
                                       self.r_disk <= self.r_max)

    @property
    def r_sky(self):
        return self.r_sky_f

    @property
    def t_sky(self):
        return self.t_sky_f

    @property
    def v0_sky(self):
        return self.v0_f

    def _check_optical_depth(self):
        """
        Set the optical depth parameters if they were not set when the class
        was instantiated.
        """
        if self.tau0 is None:
            self.tau0 = 0.0
        if self.tauq is None:
            self.tauq = self.Tbq
        if self.taueta is None:
            self.taueta = 50.
        if self.taumax is None:
            self.taumax = 100.0
        if self.r_l is None:
            self.r_l = 200.0

    def _set_linewidth(self):
        """
        Sets the Doppler linewidth profile in [m/s].
        """
        if self.dV0 is None:
            csound_f = np.sqrt(sc.k * self.Tb_f / self.mu / self.mH)
            self.dV_f = csound_f * \
                        np.sqrt(2 * self.mu / self.mu_l + self.xi_nt**2)
            self.dV_f = np.clip(self.dV_f, 0.0, self.dVmax)
            if self._flat_disk:
                self.dV_b = None
            else:
                csound_b = np.sqrt(sc.k * self.Tb_b / self.mu / self.mH)
                self.dV_b = csound_b * \
                            np.sqrt(2 * self.mu / self.mu_l + self.xi_nt**2) 
                self.dV_b = np.clip(self.dV_b, 0.0, self.dVmax)
        else:
            if self.dVq is None:
                self.dVq = 0.5 * self.Tbq
            self.dV_f = self.dV0 * (self.r_sky_f / self.r0)**(-self.dVq)
            self.dV_f = np.clip(self.dV_f, 0.0, self.dVmax)
            if self._flat_disk:
                self.dV_b = None
            else:
                self.dV_b = self.dV0 * (self.r_sky_b / self.r0)**(-self.dVq)
                self.dV_b = np.clip(self.dV_b, 0.0, self.dVmax)

    def _set_brightness(self):
        """
        Sets the brightness profile in [K].
        """
        self.Tb_f = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq) * \
                    np.exp(-(self.r_sky_f / self.r_l)**self.Tbeps)
        self.Tb_f = np.clip(self.Tb_f, 0.0, self.Tbmax)
        self.Tb_f = np.where(self._in_disk_f, self.Tb_f, 0.0)
        if self._flat_disk:
            self.Tb_b = None
        else:
            self.Tb_b = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq) * \
                        np.exp(-(self.r_sky_f / self.r_l)**self.Tbeps)
            self.Tb_b = np.clip(self.Tb_b, 0.0, self.Tbmax_b)
            self.Tb_b = np.where(self._in_disk_b, self.Tb_b, 0.0)

    def _set_rotation(self):
        """
        Sets the projected rotation profile in [m/s].
        """
        self.v0_f = self._calculate_projected_vkep(self.r_sky_f,
                                                   self.z_sky_f,
                                                   self.t_sky_f,
                                                   self.inc)
        if self._flat_disk:
            self.v0_b = None
        else:
            self.v0_b = self._calculate_projected_vkep(self.r_sky_b,
                                                       self.z_sky_b,
                                                       self.t_sky_b,
                                                       self.inc)
        return

    def _set_tau(self):
        """
        Sets the tau radial profile.
        """
        self.tau = self.tau0 * (self.r_sky_f / self.r0)**self.tauq * \
                   np.exp(-(self.r_sky_f / self.r_l)**self.taueta)
        self.tau = np.where(self._in_disk_f, self.tau, 0.0)

    def interpolate_model(self, x, y, param, x_unit='au', param_max=None,
                          interp1d_kw=None):
        """
        Interpolate a user-provided model for the brightness temperature
        profile or the line width.

        Args:
            x (array): Array of radii at which the model is sampled at in units
                given by ``x_units``, either ``'au'`` or ``'arcsec'``.
            y (array): Array of model values evaluated at ``x``. If brightness
                temperature, in units of [K], or for linewidth, units of [m/s].
            param (str): Parameter of the model, either ``'Tb'`` for brightness
                temperature, or ``'dV'`` for linewidth.
            x_unit (Optional[str]): Unit of the ``x`` array, either
                ``'au'`` or ``'arcsec'``.
            param_max (Optional[float]): If provided, use as the maximum value
                for the provided parameter (overwriting previous values).
            interp1d_kw (Optional[dict]): Dictionary of kwargs to pass to
                ``scipy.interpolate.intep1d`` used for the linear
                interpolation.
        """
        from scipy.interpolate import interp1d

        # Value the input models.

        if x.size != y.size:
            raise ValueError("`x.size` does not equal `y.size`.")
        if x_unit.lower() == 'arcsec':
            x *= self.dist
        elif x_unit.lower() != 'au':
            raise ValueError("Unknown `radii_unit` {}.".format(x_unit))
        if y[0] != 0.0 or y[-1] != 0.0:
            print("First or last value of `y` is non-zero and may cause " +
                  "issues with extrapolated values.")

        # Validate the kwargs passed to interp1d.

        ik = {} if interp1d_kw is None else interp1d_kw
        ik['bounds_error'] = ik.pop('bounds_error', False)
        ik['fill_value'] = ik.pop('fill_value', 'extrapolate')
        ik['assume_sorted'] = ik.pop('assume_sorted', False)

        # Interpolate the functions onto the coordinate grids.

        if param.lower() == 'tb':
            self.Tb_f = interp1d(x, y, **ik)(self.r_sky_f)
            self.Tb_f = np.clip(self.Tb_f, 0.0, param_max)
            if self.r_sky_b is not None:
                self.Tb_b = interp1d(x, y, **ik)(self.r_sky_b)
                self.Tb_b = np.clip(self.Tb_b, 0.0, param_max)
            self.Tb0, self.Tbq, self.Tbmax = np.nan, np.nan, param_max

        elif param.lower() == 'dv':
            self.dV_f = interp1d(x, y, **ik)(self.r_sky_f)
            self.dV_f = np.clip(self.dV_f, 0.0, param_max)
            if self.r_sky_b is not None:
                self.dV_b = interp1d(x, y, **ik)(self.r_sky_b)
                self.dV_b = np.clip(self.dV_b, 0.0, param_max)
            self.dV0, self.dVq, self.dVmax = np.nan, np.nan, param_max

        elif param.lower() == 'tau':
            self.tau = interp1d(x, y, **ik)(self.r_sky_f)
            self.tau = np.clip(self.tau, 0.0, param_max)

        else:
            raise ValueError("Unknown 'param' value {}.".format(param))

    @property
    def v0_disk(self):
        """
        Disk-frame rotation profile.
        """
        v0 = self._calculate_projected_vkep(self.r_disk, 0.0)
        return np.where(self._in_disk, v0, np.nan)

    @property
    def Tb_disk(self):
        """
        Disk-frame brightness profile.
        """
        Tb = self.Tb0 * (self.r_sky_f / self.r0)**(-self.Tbq) * \
             np.exp(-(self.r_sky_f / self.r_l)**self.Tbeps)
        return np.where(self._in_disk, Tb, np.nan)

    @property
    def dV_disk(self):
        """
        Disk-frame line-width profile.
        """
        if self.dV0 is None:
            csound = np.sqrt(sc.k * Tb_disk / self.mu / self.mH)
            dV = csound * np.sqrt(2 * self.mu / self.mu_l + self.xi_nt**2)
        else:
            if self.dVq is None:
                self.dVq = 0.5 * self.Tbq
            dV = self.dV0 * (self.r_disk / self.r0)**(-self.dVq)
        return np.where(self._in_disk, dV, np.nan)

    def _calculate_projected_vkep(self, r, z, t=0.0, inc=90.0):
        """
        Calculates the projected Keplerian rotation profile based on the
        attached stellar mass and source distance and inclination.

        Args:
            r (float/array): Cylindrical radius in [au].
            z (float/array): Cylindrical  height in [au].
            t (Optional[float/array]): Polar angle in [rad].
            inc (Optional[float]): Dist inclination in [deg].

        Returns:
            vkep (float/array): Projected Keplerian velocity in [m/s].
        """
        vkep2 = sc.G * self.mstar * self.msun * r**2.0
        vkep2 /= np.hypot(r, z)**3.0
        vkep = np.sqrt(vkep2 / sc.au)
        return vkep * np.cos(t) * abs(np.sin(np.radians(inc)))

    # -- Deprojection Functions -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, zpsi=0.0,
                    zphi=0.0, frame='cylindrical'):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile:

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi} +
            z_1 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\varphi}

        Where both ``z0`` and ``z1`` are given in [arcsec]. For a razor thin
        disk, ``z0=0.0``, while for a conical disk, ``psi=1.0``. Typically
        ``z1`` is not needed unless the data is exceptionally high SNR and well
        spatially resolved.

        It is also possible to override this parameterization and directly
        provide a user-defined ``z_func``. This allow for highly complex
        surfaces to be included. If this is provided, the other height
        parameters are ignored.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to ``z0``.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either ``'polar'`` or ``'cartesian'``.

        Returns:
            Three coordinate arrays, either the cylindrical coordaintes,
            ``(r, theta, z)`` or cartestian coordinates, ``(x, y, z)``,
            depending on ``frame``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Calculate the pixel values.

        r, t, z = self._get_flared_coords(x0, y0, inc, PA, self._z_func)
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    def _z_func(self, r):
        """
        Returns the emission height in [arcsec].
        """
        z = self.z0 * (r * self.dist / self.r0)**self.zpsi * \
            np.exp(-(r * self.dist / self.r_l)**self.zphi) / self.dist
        return np.clip(z, 0., None)

    @staticmethod
    def _rotate_coords(x, y, PA):
        """
        Rotate (x, y) by PA [deg].
        """
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
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
        x_rot, y_rot = simple_disk._rotate_coords(x_sky, y_sky, PA)
        return simple_disk._deproject_coords(x_rot, y_rot, inc)

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
    def xaxis_disk(self):
        """
        X-axis for the disk coordinates in [au].
        """
        return self.x_disk[0]

    @property
    def yaxis_disk(self):
        """
        y-axis for the disk coordinates in [au].
        """
        return self.y_disk[:, 0]

    @property
    def xaxis_sky(self):
        """
        X-axis for the sky coordinates in [arcsec].
        """
        return self.x_sky[0]

    @property
    def yaxis_sky(self):
        """
        Y-axis for the sky coordinates in [arcsec].
        """
        return self.y_sky[:, 0]

    # -- Helper Functions -- #

    def set_coordinates(self, x0=None, y0=None, inc=None, PA=None, dist=None,
                        z0=None, zpsi=None, r_min=None, r_max=None, FOV=None,
                        Npix=None):
        """
        Helper function to redefine the coordinate system.
        """
        self.x0 = self.x0 if x0 is None else x0
        self.y0 = self.y0 if y0 is None else y0
        self.inc = self.inc if inc is None else inc
        self.PA = self.PA if PA is None else PA
        self.dist = self.dist if dist is None else dist
        self.z0 = self.z0 if z0 is None else z0
        self.zpsi = self.zpsi if zpsi is None else zpsi
        self.r_min = self.r_min if r_min is None else r_min
        self.r_max = self.r_max if r_max is None else r_max
        self.FOV = self.FOV if FOV is None else FOV
        self.Npix = self.Npix if Npix is None else Npix
        self._populate_coordinates()
        self._set_brightness()
        self._set_linewidth()
        self._set_rotation()
        self._set_tau()

    def set_brightness(self, Tb0=None, Tbq=None, Tbmax=None, Tbmax_b=None):
        """
        Helper function to redefine the brightnes profile.
        """
        self.Tb0 = self.Tb0 if Tb0 is None else Tb0
        self.Tbq = self.Tbq if Tbq is None else Tbq
        self.Tbmax = self.Tbmax if Tbmax is None else Tbmax
        self.Tbmax_b = self.Tbmax_b if Tbmax_b is None else Tbmax_b
        self._set_brightness()

    def set_linewidth(self, dV0=None, dVq=None, dVmax=None):
        """
        Helper function to redefine the Doppler linewidth profile.
        """
        self.dV0 = self.dV0 if dV0 is None else dV0
        self.dVq = self.dVq if dVq is None else dVq
        self.dVmax = self.dVmax if dVmax is None else dVmax
        self._set_linewidth()

    def set_tau(self, tau0=None, tauq=None, taueta=None, r_l=None, taumax=None):
        """
        Helper function to redefine the optical depth profile.
        """
        self.tau0 = self.tau0 if tau0 is None else tau0
        self.tauq = self.tauq if tauq is None else tauq
        self.taueta = self.taueta if taueta is None else taueta
        self.taumax = self.taumax if taumax is None else taumax
        self.r_l = self.r_l if r_l is None else r_l
        self._set_tau()

    # -- Pseudo Image Functions -- #

    def get_cube(self, velax, dv0=None, bmaj=None, bmin=None, bpa=0.0, rms=0.0,
                 spectral_response=None):
        """
        Return the pseudo-cube with the given velocity axis.

        Args:
            velax (array): 1D array of channel centres in [m/s].
            dv0 (optional[ndarray]): An array of projected velocity
                perturbations in [m/s].
            bmaj (optional[float]): Synthesised beam major axis in [arcsec].
            bmin (optional[float]): Synthesised beam minor axis in [arcsec]. If
                only `bmaj` is specified, will assume a circular beam.
            bpa (optional[float]): Beam position angle in [deg].
            rms (optional[float]): RMS of the noise to add to the image.
            spectral_response (optional[list]): The kernel to convolve the cube
                with along the spectral dimension to simulation the spectral
                response of the telescope.

        Returns:
            cube (array): A 3D image cube.
        """

        # Make the image cube.

        cube = np.array([self.get_channel(velax[i], dv0=dv0)
                         for i in range(velax.size)])
        assert cube.shape[0] == velax.size, "not all channels created"

        # Include convolution.

        beam = self._get_beam(bmaj, bmin, bpa) if bmaj is not None else None
        if beam is not None:
            cube = simple_disk._convolve_cube(cube, beam)
        if spectral_response is not None:
            cube = np.convolve(cube, spectral_response, axis=0)

        # Add noise and return.

        if rms > 0.0:
            noise = np.random.randn(cube.size).reshape(cube.shape)
            if beam is not None:
                noise = simple_disk._convolve_cube(noise, beam)
            if spectral_response is not None:
                noise = np.convolve(noise, spectral_response, axis=0)
            noise *= rms / np.std(noise)
        else:
            noise = np.zeros(cube.shape)
        return cube + noise

    def get_channel(self, velax, dv0=None, bmaj=None, bmin=None,
                    bpa=0.0, rms=0.0):
        """
        Calculate the channel emission in [K]. Can include velocity
        perturbations with the `dv0` parameter. To simulate observations this
        can include convolution with a 2D Gaussian beam or the addition of
        (correlated) noise.

        Args:
            v_min (float): The minimum velocity of the channel in [m/s].
            v_max (float): The maximum velocity of the channel in [m/s].
            dv0 (optional[ndarray]): An array of projected velocity
                perturbations in [m/s].
            bmaj (optional[float]): Synthesised beam major axis in [arcsec].
            bmin (optional[float]): Synthesised beam minor axis in [arcsec]. If
                only `bmaj` is specified, will assume a circular beam.
            bpa (optional[float]): Beam position angle in [deg].
            rms (optional[float]): RMS of the noise to add to the image.

        Returns:
            channel (ndarray): A synthesied channel map in [K].
        """

        # Check to see if there are one or two perturbations provided.

        try:
            dv0_f, dv0_b = dv0
        except ValueError:
            dv0_f = dv0
        except TypeError:
            dv0_f = np.zeros(self.r_sky_f.shape)
            dv0_b = dv0_f.copy()

        # Calculate the flux from the front side of the disk.

        flux_f = self._calc_flux(velax, dv0_f, 'f')

        # If `z0 != 0.0`, can combine the front and far sides based on a
        # two-slab approximation.

        if not self._flat_disk:
            flux_b = self._calc_flux(velax, dv0_b, 'b')
            frac_f, frac_b = self._calc_frac(velax, dv0_b)
            flux = frac_f * flux_f + frac_b * flux_b
        else:
            flux = flux_f

        # Include a beam convolution if necessary.

        beam = None if bmaj is None else self._get_beam(bmaj, bmin, bpa)
        if beam is not None:
            flux = convolve(flux, beam)

        # Add noise and return.

        noise = np.random.randn(flux.size).reshape(flux.shape)
        if beam is not None:
            noise = convolve(noise, beam)
        noise *= rms / np.std(noise)
        return flux + noise

    def get_channel_tau(self, velax, dv0=0.0, bmaj=None, bmin=None, bpa=0.0):
        """
        As ``get_channel``, but returns the optical depth of the front side of
        the disk.

        Args:
            v_min (float): The minimum velocity of the channel in [m/s].
            v_max (float): The maximum velocity of the channel in [m/s].
            dv0 (optional[ndarray]): An array of projected velocity
                perturbations in [m/s].
            bmaj (optional[float]): Synthesised beam major axis in [arcsec].
            bmin (optional[float]): Synthesised beam minor axis in [arcsec]. If
                only `bmaj` is specified, will assume a circular beam.
            bpa (optional[float]): Beam position angle in [deg].

        Returns:
            channel (ndarray): A synthesied channel map representing the
                optical depth.
        """

        # Calculate the optical depth.

        tau = self._calc_tau(velax, dv0=dv0)

        # Include a beam convolution if necessary.

        beam = None if bmaj is None else self._get_beam(bmaj, bmin, bpa)
        if beam is not None:
            tau = convolve(tau, beam)
        return tau

    def _calc_tau(self, velax, dv0=0.0):
        """
        Calculate the average tau profile assuming a single Gaussian component.
        """
        tau, dV, v0 = self.tau, self.dV_f, self.v0_f + dv0
        optdepth = np.empty_like(tau)
        ok = (tau > 0.)
        optdepth[~ok] = 0.
        optdepth[ok] = tau[ok] * np.exp(-((velax - v0[ok]) / dV[ok])**2) 
        return optdepth

    def _calc_flux(self, velax, dv0=0.0, side='f'):
        """
        Calculate the emergent flux assuming single Gaussian component.
        """
        if side.lower() == 'f':
            Tb, dV, v0 = self.Tb_f, self.dV_f, self.v0_f + dv0
        elif side.lower() == 'b':
            Tb, dV, v0 = self.Tb_b, self.dV_b, self.v0_b + dv0
        else:
            quote = "Unknown 'side' value {}. Must be 'f' or 'r'."
            raise ValueError(quote.format(side))
        spec = np.empty_like(Tb)
        ok = (Tb > 0.)
        spec[~ok] = 0.
        spec[ok] = Tb[ok] * np.exp(-((velax - v0[ok]) / dV[ok])**2)
        return spec

    def _calc_frac(self, velax, dv0=0.0):
        """
        Calculates the fraction of the front side of the disk realtive to the
        back side based on the optical depth.
        """
        tau = self._calc_tau(velax, dv0=dv0)
        return 1.0, np.exp(-tau)

    @staticmethod
    def _convolve_cube(cube, beam):
        """
        Convolve the cube.
        """
        return np.array([convolve(c, beam) for c in cube])

    def _get_beam(self, bmaj, bmin=None, bpa=0.0):
        """
        Make a 2D Gaussian kernel for convolution.
        """
        bmin = bmaj if bmin is None else bmin
        bmaj /= self.cell_sky * self.fwhm
        bmin /= self.cell_sky * self.fwhm
        return Gaussian2DKernel(bmin, bmaj, np.radians(bpa))

    # -- Velocity Perturbations -- #

    def _perturbation(self, r0, t0, dr, dt=0.0, beta=0.0, projection='sky',
                      trim_values=False):
        """
        Define a velocity perturbation in cylindrical coordinates in either
        sky-plane coordaintes, ``projection='sky'``, or disk plane coordinates,
        ``projection='disk'``. If ``dt`` is set to zero, it assumes an
        azimuthally symmetric perturbation.

        Args:
            r0 (float): Radius of perturbation center. If ``projection='sky'``
                this is in [arcsec], while for ``projection='disk'`` this is in
                [au]. For elevated emission surfaces this can additionally be
                ``'f'`` for the front side, or ``'b'`` for the back side.
            t0 (float): Polar angle in [degrees] of perturbation center.
            dr (float): Radial width of perturbation. If ``projection='sky'``
                this is in [arcsec], while for ``projection='disk'`` this is in
                [au].
            dt (Optional[float]): Azimuthal extent of perturbations in [deg].
            beat (Optional[float]): Fixed pitch angle in [deg].
            projection (Optional[str]): If ``'sky'``, return the function in
                sky coordinates, otherwise in disk coordinates.
            trim_values(Optional[float]): If a number is specfied, fill all
                absolute values below this as ``np.nan``, primarily used for
                plotting.

        Returns:
            f (array): 2D array of the Gaussian perturbation.
        """

        # Parse input variables.

        if projection.lower() == 'sky' or projection.lower() == 'f':
            rvals, tvals = self.r_sky / self.dist, self.t_sky
        elif projection.lower() == 'b':
            rvals, tvals = self.r_sky_b / self.dist, self.t_sky_b
        elif projection.lower() == 'disk':
            rvals, tvals = self.r_disk, self.t_disk
        else:
            raise ValueError("`projection` must be 'sky', 'f', 'b' or 'disk'.")
        if dt == 0.0 and beta != 0.0:
            raise ValueError("Cannot specify pitch angle and `dt=0.0`.")

        # Azimuthally symmetric perturbation.

        if dt == 0.0:
            return np.exp(-0.5*((rvals - r0) / dr)**2.0)

        # Calculate azmithal dependance.

        f = []
        nwrap = self.nwrap if self.nwrap % 2 else self.nwrap + 1
        for wrap in np.arange(nwrap) - (nwrap - 1) / 2:

            t_tmp = tvals.copy() + wrap * 2.0 * np.pi
            r0_tmp = r0 / (1.0 + t_tmp * np.tan(np.radians(beta)))
            t_tmp -= np.radians(t0)

            _f = np.exp(-0.5*((rvals - r0_tmp) / dr)**2.0)
            f += [_f * np.exp(-0.5*(t_tmp / np.radians(dt))**2.0)]
        f = np.sum(f, axis=0)

        # Apply trims and return.

        if trim_values:
            f = np.where(abs(f) > trim_values, f, np.nan)
        return f

    def radial_perturbation(self, dv, r0, t0, dr, dt=0.0, beta=0.0,
                            projection='sky', trim_values=False):
        """
        Gaussian perturbation with radial velocity projection.
        """
        f = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt, beta=beta,
                                    projection=projection,
                                    trim_values=trim_values)
        if projection.lower() == 'disk':
            return f
        return f * np.sin(self.t_sky) * np.sin(np.radians(self.inc))

    def rotational_perturbation(self, dv, r0, t0, dr, dt=0.0, beta=0.0,
                                projection='sky', trim_values=False):
        """
        Gaussian perturbation with rotational velocity projection.
        """

        # Disk projection.

        if projection.lower() == 'disk':
            return dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt,
                                           beta=beta, projection='disk',
                                           trim_values=trim_values)

        elif not projection.lower() == 'sky':
            raise ValueError("'projection' must be 'sky' or 'disk'.")

        # If a sky projection, check to see if two sides are needed.

        f = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt,
                                    beta=beta, projection='f',
                                    trim_values=trim_values)
        f *= np.cos(self.t_sky_f) * np.sin(np.radians(self.inc))

        if self._flat_disk:
            return f

        b = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt,
                                    beta=beta, projection='b',
                                    trim_values=trim_values)
        b *= np.cos(self.t_sky_f) * np.sin(np.radians(self.inc))

        return f, b

    def vertical_perturbation(self, dv, r0, t0, dr, dt=0.0, beta=0.0,
                              projection='sky', trim_values=False):
        """
        Gaussian perturbation with vertical velocity projection.
        """
        f = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt, beta=beta,
                                    projection=projection,
                                    trim_values=trim_values)
        if projection.lower() == 'disk':
            return f
        return f * np.cos(self.inc)

    def doppler_flip(self, dv, r0, t0, dr, dt, beta=0.0, dr0=0.5, dt0=1.0,
                     clockwise=True, projection='sky', trim_values=False):
        """
        Simple 'Doppler flip' model with two offset azimuthal deviations.

        Args:
            v (float): Azimuthal velocity deviation in [m/s].
            r0 (float): Radius in [au] of Doppler flip center.
            t0 (float): Polar angle in [degrees] of Doppler flip center.
            dr (float): Radial width of each Gaussian in [au].
            dt (float): Azimuthal width (arc length) of each Gaussian in [au].
            dr0 (Optional[float]): Relative radial offset between the positive
                and negative lobes. Defaults to 0.5.
            dt0 (Optional[float]): Relative azimuthal offset between the
                positive and negative lobes. Defaults to 1.0.

        Returns:
            dv0 (array): Array of velocity devitiations in [m/s]. If
                ``projection='sky'``, these will be projected on the sky.
        """
        rp = r0 + dr0 * dr
        rn = r0 - dr0 * dr
        dt0 /= self.dist if projection.lower() == 'sky' else 1.0
        tp = t0 + np.degrees(dt0 * dt / rp)
        tn = t0 - np.degrees(dt0 * dt / rn)
        if not clockwise:
            temp = tn
            tn = tp
            tp = temp
            beta = -beta
        vp = self.rotational_perturbation(dv=dv, r0=rp, t0=tp, dr=dr, dt=dt,
                                          beta=-beta, projection=projection)
        vn = self.rotational_perturbation(dv=dv, r0=rn, t0=tn, dr=dr, dt=dt,
                                          beta=-beta, projection=projection)
        v = vp - vn
        if trim_values:
            v = np.where(abs(v) > trim_values, v, np.nan)
        return v

    def radial_doppler_flip(self, dv, r0, t0, dr, dt, dr0=0.5, dt0=1.0,
                            flip_rotation=False, projection='sky',
                            trim_values=False):
        """
        Simple `Doppler flip` model but with radial velocity deviations intead.

        Args:
            dv (float): Radial velocity deviation in [m/s].
            r0 (float): Radius in [au] of Doppler flip center.
            t0 (float): Polar angle in [degrees] of Doppler flip center.
            dr (float): Radial width of each Gaussian in [au].
            dt (float): Azimuthal width (arc length) of each Gaussian in [au].
            dr0 (Optional[float]): Relative radial offset between the positive
                and negative lobes. Defaults to 0.5.
            dt0 (Optional[float]): Relative azimuthal offset between the
                positive and negative lobes. Defaults to 1.0.

        Returns:
            dv0 (array): Array of velocity devitiations in [m/s]. If
                ``sky=True``, these will be projected on the sky.
        """
        rp = r0 + dr0 * dr
        rn = r0 - dr0 * dr
        tp = t0 + np.degrees(dt0 * dt / rp)
        tn = t0 - np.degrees(dt0 * dt / rn)
        if flip_rotation:
            temp = tn
            tn = tp
            tp = temp
        vp = self.radial_perturbation(dv=dv, r0=rp, t0=tp, dr=dr, dt=dt,
                                      projection=projection)
        vn = self.radial_perturbation(dv=dv, r0=rn, t0=tn, dr=dr, dt=dt,
                                      projection=projection)
        v = vp - vn
        if trim_values:
            v = np.where(abs(v) > trim_values, v, np.nan)
        return v

    def vertical_flow(self, v, r0, t0, dr, dt):
        raise NotImplementedError("Coming soon. Maybe.")
        return

    # -- Plotting Routines -- #

    def plot_keplerian(self, fig=None, logy=True, top_axis=True):
        """
        Plot the Keplerian rotation profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_disk.flatten()
        y = self.v0_disk.flatten()
        idxs = np.argsort(x)
        ax.plot(x[idxs], y[idxs])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Keplerian Rotation [m/s]')
        if logy:
            ax.set_yscale('log')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig

    def plot_linewidth(self, fig=None, top_axis=True):
        """
        Plot the linewidth profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky_f.flatten()
        y = self.dV_f.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]
        mask = np.logical_and(x >= self.r_min, x <= self.r_max)
        ax.plot(x[mask], y[mask])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Doppler Linewidth [m/s]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig

    def plot_brightness(self, fig=None, top_axis=True):
        """
        Plot the brightness temperature profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky_f.flatten()
        y = self.Tb_f.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]
        mask = np.logical_and(x >= self.r_min, x <= self.r_max)
        ax.plot(x[mask], y[mask])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('BrightestTemperature [K]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig

    def plot_tau(self, fig=None, top_axis=True):
        """
        Plot the optical depth profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky_f.flatten()
        y = self.tau.flatten()
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]
        mask = np.logical_and(x >= self.r_min, x <= self.r_max)
        ax.plot(x[mask], y[mask])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Optical Depth')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig

    def plot_emission_surface(self, fig=None, top_axis=True):
        """
        Plot the emission surface.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky_f.flatten()
        y = self._z_func(x / self.dist) * self.dist
        idxs = np.argsort(x)
        x, y = x[idxs], y[idxs]
        mask = np.logical_and(x >= self.r_min, x <= self.r_max)
        ax.plot(x[mask], y[mask])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Emission Height [au]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')
        return fig

    def plot_radii(self, ax, rvals, contour_kwargs=None, projection='sky',
                   side='f'):
        """
        Plot annular contours onto the axis.
        """
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', '0.6')
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 0.5)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '--')
        if projection.lower() == 'sky':
            if 'f' in side:
                r = self.r_sky_f
            elif 'b' in side:
                r = self.r_sky_b
            else:
                raise ValueError("Unknown 'side' value {}.".format(side))
            x, y, z = self.x_sky[0], self.y_sky[:, 0], r / self.dist
        elif projection.lower() == 'disk':
            x, y, z = self.x_disk, self.y_disk, self.r_disk
        ax.contour(x, y, z, rvals, **contour_kwargs)

    @staticmethod
    def format_sky_plot(ax):
        """
        Default formatting for sky image.
        """
        from matplotlib.ticker import MaxNLocator
        ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Offset [arcsec]')
        ax.set_ylabel('Offset [arcsec]')
        ax.scatter(0, 0, marker='x', color='0.7', lw=1.0, s=4)

    @staticmethod
    def format_disk_plot(ax):
        """
        Default formatting for disk image.
        """
        from matplotlib.ticker import MaxNLocator
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.set_xlabel('Offset [au]')
        ax.set_ylabel('Offset [au]')
        ax.scatter(0, 0, marker='x', color='0.7', lw=1.0, s=4)

    @staticmethod
    def BuRd():
        """Blue-Red color map."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))
        return mcolors.LinearSegmentedColormap.from_list('BuRd', colors)

    @staticmethod
    def RdBu():
        """Red-Blue color map."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))[::-1]
        return mcolors.LinearSegmentedColormap.from_list('RdBu', colors)

    @property
    def extent_sky(self):
        return [self.x_sky[0, 0],
                self.x_sky[0, -1],
                self.y_sky[0, 0],
                self.y_sky[-1, 0]]

    @property
    def extent_disk(self):
        return [self.x_sky[0, 0] * self.dist,
                self.x_sky[0, -1] * self.dist,
                self.y_sky[0, 0] * self.dist,
                self.y_sky[-1, 0] * self.dist]
