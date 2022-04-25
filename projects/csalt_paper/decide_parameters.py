import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from parametric_disk_CSALT import parametric_disk as pardisk_csalt
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from csalt.models import cube_to_fits
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)
import cmasher as cmr
from gofish import imagecube
from scipy.integrate import trapz, cumtrapz
from scipy import interpolate


# set velocities
velax = np.arange(-10000 + 5000, 10000 + 5000, 500.)

# load configs
inpc = importlib.import_module('gen_fiducial_std')
inpr = importlib.import_module('gen_radmc_std')

# calculate cubes
fixedc = inpc.nu_rest, inpc.FOV[0], inpc.Npix[0], inpc.dist, inpc.cfg_dict
fixedr = inpr.nu_rest, inpr.FOV[0], inpr.Npix[0], inpr.dist, inpr.cfg_dict
cubec = pardisk_csalt(velax, inpc.pars, fixedc)
cuber = pardisk_radmc(velax, inpr.pars, fixedr)


# create FITS
cube_to_fits(cubec, 'cube_csalt.fits', RA=240., DEC=-40.)
cube_to_fits(cuber, 'cube_radmc.fits', RA=240., DEC=-40.)


# generate and load 0th moment map (made with bettermoments)
os.system('bettermoments cube_radmc.fits -method zeroth -clip 0')
cuber = imagecube('cube_radmc_M0.fits')
hdr = fits.open('cube_radmc_M0.fits')[0].header
dxr = 3600 * hdr['CDELT1'] * (np.arange(hdr['NAXIS1']) - (hdr['CRPIX1'] - 1))
dyr = 3600 * hdr['CDELT2'] * (np.arange(hdr['NAXIS2']) - (hdr['CRPIX1'] - 1))
bmr = np.abs(np.diff(dxr)[0] * np.diff(dyr)[0])

# extract radial profile 
xr, yr, dyr = cuber.radial_profile(inc=inpr.pars[0], PA=inpr.pars[1], 
                                   x0=0.0, y0=0.0, PA_min=90, PA_max=270, 
                                   abs_PA=True, exclude_PA=False)
         
# convert to brightness units of Jy * km/s / arcsec**2
yr /= 1000       # Jy m/s / pixel to Jy km/s / pixel
yr /= bmr

# generate and load 0th moment map (made with bettermoments)
os.system('bettermoments cube_csalt.fits -method zeroth -clip 0')
cubec = imagecube('cube_csalt_M0.fits')
hdc = fits.open('cube_csalt_M0.fits')[0].header
dxc = 3600 * hdc['CDELT1'] * (np.arange(hdc['NAXIS1']) - (hdc['CRPIX1'] - 1))
dyc = 3600 * hdc['CDELT2'] * (np.arange(hdc['NAXIS2']) - (hdc['CRPIX1'] - 1))
bmc = np.abs(np.diff(dxc)[0] * np.diff(dyc)[0])

# extract radial profile 
xc, yc, dyc = cubec.radial_profile(inc=inpc.pars[0], PA=inpc.pars[1],
                                   x0=0.0, y0=0.0, PA_min=90, PA_max=270,
                                   abs_PA=True, exclude_PA=False)

# convert to brightness units of Jy * km/s / arcsec**2
yc /= 1000       # Jy m/s / pixel to Jy km/s / pixel
yc /= bmc


# integrated flux profile
def fraction_curve(radius, intensity):
    
    intensity[np.isnan(intensity)] = 0 
    total = trapz(2 * np.pi * radius * intensity, radius)   
    cum = cumtrapz(2 * np.pi * radius * intensity, radius)
    
    return cum/total, total


# size interpolator
def Reff_fraction_smooth(radius, intensity, fraction=0.95):

    curve, flux = fraction_curve(radius, intensity)
    curve_smooth = interpolate.interp1d(curve, radius[1:])

#    plt.plot(radius[1:], curve)
#    plt.plot(curve_smooth(fraction), fraction, 'o')
#    plt.show()

    return curve_smooth(fraction), flux


# get the size and the flux
sizer, fluxr = Reff_fraction_smooth(xr, yr * np.cos(np.radians(inpr.pars[0])), 
                                    fraction=0.9)
sizec, fluxc = Reff_fraction_smooth(xc, yc * np.cos(np.radians(inpc.pars[0])), 
                                    fraction=0.9)

# return the size
print('\n\n RADMC: ')
print("CO integrated flux = %f Jy km/s" % fluxr)
print("CO effective radius = %f au" % (inpr.dist * sizer))

print('\n\n CSALT: ')
print("CO integrated flux = %f Jy km/s" % fluxc)
print("CO effective radius = %f au" % (inpc.dist * sizec))
