"""
Task to build a Keplerian mask based on imaging parameters. This will read in
the image properties from an initial imaging run which outputs a dirty image.

Usage
=====

Load up the functions.

> execfile('path/to/keplerian_mask.py')
> Succesfully imported `make_mask`.

With this loaded, to make a Keplerian mask you will get,

> make_mask('image_name.image', inc=30.0, PA=75.0,
>           mstar=1.0, dist=140.0, vlsr=5.1e3)

which will produce an 'image_name.mask.image' mask.

Additional Parameters
=====================

We can also include a non-zero emission height for molecules like 12CO. This
can either by specified by a constant z/r value with the `zr` argument,

> make_mask('image_name.image', inc=30.0, PA=75.0,
>           mstar=1.0, dist=140.0, vlsr=5.1e3, zr=0.3)

If you want a more complex emission surface you can define a function which
take the midplane radius in arcseconds and returns the emission height in
arcseconds,

> def z_func(r):
>     return 0.3 * r**1.5
>
> make_mask('image_name.image', inc=30.0, PA=75.0,
>           mstar=1.0, dist=140.0, vlsr=5.1e3,
>           z_func=z_func)

In addition to the emission surface, a radial profile for the line width must
be considered. This is assumed to be a simple powerlaw,

dV(r) = dV0 * (r / 1")**dVq

which can be changed with the `dV0` and `dVq` parameters.

> make_mask('image_name.image', inc=30.0, PA=75.0,
>           mstar=1.0, dist=140.0, vlsr=5.1e3,
>           dV0=500.0, dVq=-0.45)

Note that these will have a significant effect on the shape of the mask,
particularly in the outer disk.

Finally, one can also convolve the mask with a 2D Gaussian beam. This can
either be a scale version of the clean beam attached to the image, using the
parameter `nbeams`,

> make_mask('image_name.image', inc=30.0, PA=75.0,
>           mstar=1.0, dist=140.0, vlsr=5.1e3, nbeams=1.5)

or specify the FWHM in arcseconds of a circular convolution kernel with
`target_res`,

> make_mask('image_name.image', inc=30.0, PA=75.0,
>           mstar=1.0, dist=140.0, vlsr=5.1e3, target_res=1.0)

Author
======

Written by Richard Teague, 2020.
richard.d.teague@cfa.harvard.edu
"""

import os
import numpy as np
import scipy.constants as sc
from casatasks import (imhead, rmtables, imsmooth, makemask, exportfits, imstat)
import casatools


def _get_axis_idx(header, axis_name):
    """Return the axis number of the given axis."""
    axes = ['right ascension', 'declination', 'stokes', 'frequency']
    assert axis_name.lower() in axes, "Unknown `axis_name`."
    for ax in range(1, len(axes)+1):
        key = 'ctype{:d}'.format(ax)
        if header[key].lower() == axis_name.lower():
            return ax
    raise ValueError("Cannot find requested axis in the image.")


def _string_to_Hz(string):
    """Convert a string to a frquency in [Hz]."""
    if isinstance(string, float):
        return string
    if isinstance(string, int):
        return string
    factor = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1e0}
    raw = string
    for key in ['GHz', 'MHz', 'kHz', 'Hz']:
        raw = raw.replace(key, '')
    for key in ['GHz', 'MHz', 'kHz', 'Hz']:
        if key in string:
            return float(raw) * factor[key]


def _get_offsets(image, restfreqs=None):
    """Convert rest frequencies in [Hz] to velocity offsets in [m/s]."""
    header = imhead(image, mode='list')

    # Make an iterable list of frequencies.
    if restfreqs is None:
        restfreqs = header['restfreq']
    restfreqs = np.atleast_1d(restfreqs)

    # Frequency axis.
    offsets = []
    freq = _make_axis(header, 'frequency')
    velo = _make_axis(header, 'velocity')
    for restfreq in restfreqs:
        nu = _string_to_Hz(restfreq)
        temp = sc.c * (nu - freq) / nu
        offsets += [np.mean(velo - temp)]
    return offsets


def _make_axis(header, axis_name):
    """
    Make the requestest axis based on the provided image. Assumes that the disk
    is centered. TODO: Check half-pixel offset.

    Args:
        header (CASA image header): Image header of the image you want to make
            a mask for.
        axis_name (str): The axis name to generate. Must be one of: 'velocity',
            'frequency', 'right ascension' or 'declination'.

    Returns:
        axis (ndarray): The requested axis.
    """

    # If we want the velocity axis, make a frequnecy
    # axis first and then convert to velocity.
    if axis_name.lower() == 'velocity':
        axis_name = 'frequency'
        convert_frequency = True
    else:
        convert_frequency = False

    # Read in the parameters for the axis.
    idx = _get_axis_idx(header, axis_name)
    alen = header['shape'][idx - 1]
    apix = header['crpix{:d}'.format(idx)]
    adel = header['cdelt{:d}'.format(idx)]
    aref = header['crval{:d}'.format(idx)] - 1

    # Correct the values based on the axis generated.
    if axis_name.lower() == 'right ascension':
        adel *= np.cos(adel * np.pi / 2.0)
    if axis_name.lower() in ['right ascension', 'declination']:
        if header['cunit{:d}'.format(idx)] == 'rad':
            adel = np.degrees(adel)
        elif header['cunit{:d}'.format(idx)] != 'deg':
            raise ValueError("Unknown spatial axis unit.")
        adel *= 3600.
        aref, apix = 0.0, (alen / 2) - 0.5
    axis = aref + (np.arange(alen) - apix) * adel
    if convert_frequency:
        rest = header['restfreq']
        axis = (rest - axis) * sc.c / rest
    return axis


def _generate_axes(image):
    """
    Reads an image header to create the axes which match the provided image.

    Args:
        image (str): Image to generate the axes of.

    Returns:
        xaxis, yaxis, saxis, vaxis (ndarrays): Right ascension, declination,
            Stokes and velocity axes of the image. If there is no attached
            Stokes axis will return a single valued array.
    """
    header = imhead(image, mode='list')
    xaxis = _make_axis(header, 'right ascension')
    yaxis = _make_axis(header, 'declination')
    try:
        saxis = _make_axis(header, 'stokes')
    except KeyError:
        saxis = np.zeros(1)
    vaxis = _make_axis(header, 'velocity')
    return xaxis, yaxis, saxis, vaxis


def _deproject(x, y, dx0=0.0, dy0=0.0, inc=0.0, PA=0.0, zr=0.0, z_func=None):
    """
    Deproject the data based on the source properties.

    Args:
        x (ndarray): Sky plane right ascension coordinates in [arcsec].
        y (ndarray): Sky plane declination coordinates in [arcsec].
        dx0 (optional[float]): Offset in right ascension for the source center
            in [arcsec].
        dy0 (optional[float]): Offset in declination for the source center in
            [arcsec].
        inc (optional[float]): Disk inclination in [deg].
        PA (optional[float]): Disk position angle, measured to the redshifted
            axis in an Eastward direction in [deg].
        zr (optional[float]): z/r value to assume for the emission.
        z_func (optional[callabe]): A user-defined emission height function
            returning the height of the emission in [arcsec] for a given radius
            in [arcsec].

    Returns:
        rvals, tvals, zvals (ndarrays): Radius, azimuthal and height
            deprojected coordinates in [arcsec], [rad], [arcec], respectively.
    """

    # Define the emission function. This is bit messy to account for the
    # possibility of both user-defined emission surfaces and a simple conical
    # surface.
    if z_func is None:
        def z_func_tmp(r):
            return zr * r
    else:
        def z_func_tmp(r):
            return zr * z_func(r)
        assert callable(z_func), "Must provide a callable `z_func`."

    # Iterate to define the correct correction for the height.
    x_mid, y_mid = _midplane_coords(x, y, dx0, dy0, inc, PA)
    r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
    for _ in range(10):
        z_tmp = z_func_tmp(r_tmp)
        y_tmp = y_mid + z_tmp * np.tan(np.radians(inc))
        r_tmp = np.hypot(y_tmp, x_mid)
        t_tmp = np.arctan2(y_tmp, x_mid)
    return (r_tmp.T, t_tmp.T, z_tmp.T)


def _rotate(x, y, PA):
    """Rotate the cartesian coordinates by PA in [deg]."""
    x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
    y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
    return x_rot, y_rot


def _incline(x, y, inc):
    """Incline the cartesian coordinates by inc in [deg]."""
    x_inc = x
    y_inc = y / np.cos(np.radians(inc))
    return x_inc, y_inc


def _midplane_coords(x, y, dx0=0.0, dy0=0.0, inc=0.0, PA=0.0):
    """Get the midplane cartesian coordiantes."""
    x_mid, y_mid = np.meshgrid(x - dx0, y - dy0)
    x_mid, y_mid = _rotate(x_mid, y_mid, PA)
    return _incline(x_mid, y_mid, inc)


def _keplerian(r, t, z, mstar, dist, inc):
    """Calculate projected Keplerian rotation at each pixel."""
    v = sc.G * mstar * 1.989e30 * (r * dist * sc.au)**2
    v *= np.power(np.hypot(r, z) * sc.au * dist, -3.0)
    return np.sqrt(v) * np.cos(t) * np.sin(np.radians(abs(inc)))


def _get_disk_coords(image, dx0, dy0, inc, PA, zr, z_func):
    """Return the deprojected disk cylindrical coordinates."""
    x, y, s, v = _generate_axes(image)
    rvals, tvals, zvals = _deproject(x=x, y=y, dx0=dx0, dy0=dy0, inc=inc,
                                     PA=PA, zr=zr, z_func=z_func)
    rvals = rvals[:, :, None, None] * np.ones((x.size, y.size, s.size, v.size))
    tvals = tvals[:, :, None, None] * np.ones((x.size, y.size, s.size, v.size))
    zvals = zvals[:, :, None, None] * np.ones((x.size, y.size, s.size, v.size))
    return rvals, tvals, zvals


def _get_projected_vkep(rvals, tvals, zvals, mstar, dist, inc, vlsr):
    """Get the projected Keplerian rotation in [m/s]."""
    return _keplerian(rvals, tvals, zvals, mstar, dist, inc) + vlsr


def _get_linewidth(rvals, dV0, dVq):
    """Return the Doppler width in [m/s] of the line at each position."""
    return dV0 * rvals**dVq


def _trim_name(image):
    """Remove the slash at the end of the filename."""
    return image[:-1] if image[-1] == '/' else image


def _save_as_image(image, mask, overwrite=True):
    """Save as an image by copying the header info from 'image'."""
    ia = casatools.image()
    ia.open(image)
    coord_sys = ia.coordsys().torecord()
    ia.close()
    outfile = _trim_name(image).replace('.image', '.mask.image')
    if overwrite:
        rmtables(outfile)
    ia.fromarray(pixels=mask, outfile=outfile, csys=coord_sys)
    ia.close()
    ia.done()


def _read_beam(image, axis='major'):
    """Read the beam size. Can handle beam tables if present."""
    header = imhead(image, mode='list')
    try:
        beam = header['perplanebeams']['median area beam']
        return beam[axis]['value']
    except KeyError:
        axis = 'pa' if axis == 'positionangle' else axis
    return header['beam{}'.format(axis)]['value']


def _convolve_image(image, mask, nbeams=None, target_res=None, overwrite=True):
    """
    Convolve the mask with a 2D Gaussian beam.

    Args:
        image (str): Path to the image to containing the beam to use.
        mask (str): Path to the mask to convolve.
        nbeams (optional[float]): Scale the convolution kernel to this many
            times the clean beam size of the image.
        target_res (optional[float]): Size of the convolution kernel in arcsec.
        overwrite (optional[bool]): If True, overwrite the input image with
            the convolved image.
    """
    image = image[:-1] if image[-1] == '/' else image
    if nbeams is None and target_res is None:
        raise ValueError("Must specify 'nbeams' or 'target_res'.")
    if target_res is None:
        major = _read_beam(image, 'major') * nbeams
        minor = _read_beam(image, 'minor') * nbeams
    else:
        major = target_res
        minor = target_res
    if isinstance(major, float):
        major = '{:.2f}arcsec'.format(major)
        minor = '{:.2f}arcsec'.format(minor)
    imsmooth(imagename=mask, outfile=mask+'.conv',
             overwrite=True, kernel='gauss', major=major, minor=minor,
             pa='{:.2f}deg'.format(_read_beam(image, 'positionangle')))
    if overwrite:
        os.system('rm -rf {}'.format(mask))
        os.system('mv {}.conv {}'.format(mask, mask))


def _make_zr_list(zr, max_dzr=0.1):
    """List of equally spaced z/r heights with a minimum spacing max_dzr."""
    if zr == 0.0:
        return np.zeros(1)
    a = np.arange(0.0, zr, max_dzr)
    a = np.append(a, zr) if a[-1] != zr else a
    a = np.concatenate([-a[1:][::-1], a])
    return np.linspace(a[0], a[-1], a.size)


def _save_as_mask(image, tolerance=0.01):
    """
    Save the provided images as a boolean mask.

    Args:
        image (str): Image to save as a mask.
        tolerance (optional[float]): Values below this value considered to be
            masked.
    """
    ia = casatools.image()
    ia.open(image)
    ia.calcmask('"{}" > {:.2f}'.format(image, tolerance), name='mask0')
    ia.close()
    ia.done()
    makemask(mode='copy', inpimage=image, inpmask='{}:mask0'.format(image),
             output=image, overwrite=True)


def make_kepmask(image, inc, PA, dist, mstar, vlsr, dx0=0.0, dy0=0.0, zr=0.0,
                 z_func=None, dV0=300.0, dVq=-0.5, r_min=0.0, r_max=4.0,
                 nbeams=None, target_res=None, tolerance=0.01, restfreqs=None,
                 estimate_rms=False, max_dzr=0.2, export_FITS=False):
    """
    Make a Keplerian mask for CLEANing.

    Args:
        image (str): Path to the image file to make the mask for.
        inc (float): Inclination of the disk in [deg].
        PA (float): Position angle of the disk, measured Eastwards of North to
            the redshifted axis, in [deg].
        dist (float): Source distance in [pc].
        mstar (float): Mass of the central star in [Msun].
        vlsr (float): Systemic velocity in [m/s].
        disk_dict (dict): Dictionary of disk parameters from
            'generalanalysis/diskdictionary.py'.
        dx0 (optional[float]): Source center offset along x-axis [arcsec].
        dy0 (optional[float]): Source center offset along y-axis [arcsec].
        zr (optional[float]): For elevated emission, the z/r value.
        z_func (optional[callable]: For elevated emission, a callable
            function which takes the disk midplane radius in [arcsec] and
            returns the emission height in [arcsec]. This will take precedent
            over `zr`.
        dV0 (optional[float]): The Doppler width of the line in [m/s] at 1
            arcsec.
        dVq (optional[float]): The exponent of the power law describing the
            Doppler width as a function of radius.
        r_min (optional[float]): Minimum radius in [arcsec] of the mask.
        r_max (optional[float]): Maximum radius in [arcsec] of the mask.
        nbeams (optional[float]): Convovle the mask with a beam with axes
            scaled by a factor of `nbeams`.
        target_res (optional[float]): Instead of scaling the CLEAN beam for the
            convolution kernel, specify the FWHM of the convolution kernel
            directly.
        tolerance (optional[float]): The threshold to consider the convolved
            mask where there is emisson. Typically used to remove the noise
            from the convolution.
        restfreqs (optional[list]): If the image contains multiple lines, a
            list of their rest frequencies. Can either be in strings
            including the unit, ``'230.580GHz'``, or as floats, ``230.580e9``,
            assumed to be in [Hz].
        estimate_rms (optional[bool]): If True, calculate and return the RMS of
            the masked regions to estimate CLEANing thresholds.
        max_dzr (optional[float]): Maximum spacing in zr to use when filling in
            the image plane for highly elevated models.
        export_FITS (optional[bool]): If True, export the mask as a FITS file.

    Returns:
        rms (float): The RMS of the masked regions if `estimate_rms` is True.
    """
    # Grab the velocity axis.
    image = image if image[-1] != '/' else image[:-1]
    v_axis = _generate_axes(image)[-1]
    dvchan = 0.5 * abs(np.diff(v_axis).mean())

    # Define the rest frequencies and cycle through them.
    mask = None
    zr_list = _make_zr_list(zr, max_dzr) if z_func is None else [-1., 0., 1.]
    for offset in _get_offsets(image, restfreqs):
        for zr in zr_list:
            r, t, z = _get_disk_coords(image, dx0, dy0, inc, PA, zr, z_func)
            vkep = _get_projected_vkep(r, t, z, mstar, dist, inc, vlsr+offset)
            dV = _get_linewidth(r, dV0, dVq)
            r_mask = np.logical_and(r >= r_min, r <= r_max)
            v_mask = abs(v_axis[None, None, None, :] - vkep) < dV + dvchan
            tmp_mask = np.logical_and(r_mask, v_mask)
            if mask is None:
                mask = np.where(tmp_mask, 1.0, 0.0)
            else:
                mask = np.where(np.logical_or(mask, tmp_mask), 1.0, 0.0)

    # Save it as a mask. Again, clunky but it works.
    _save_as_image(image, mask)
    if (nbeams is not None) or (target_res is not None):
        _convolve_image(image, image.replace('.image', '.mask.image'),
                        nbeams=nbeams, target_res=target_res)
    _save_as_mask(image.replace('.image', '.mask.image'), tolerance)
    mask = image.replace('.image', '.mask.image')

    # Export as a FITS file if requested.
    if export_FITS:
        exportfits(imagename=mask, fitsimage=mask.replace('.image', '.fits'),
                   dropstokes=True)

    # Estimate the RMS of the un-masked pixels.
    if estimate_rms:
        rms = imstat(imagename=image, mask='"{}" < 1.0'.format(mask))['rms'][0]
        print_rms = rms if rms > 1e-2 else rms * 1e3
        print_unit = 'Jy' if rms > 1e-2 else 'mJy'
        print("# Estimated RMS of unmasked regions: " +
              "{:.2f} {}/beam".format(print_rms, print_unit))
        return rms

    return None
