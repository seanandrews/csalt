import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from csalt.models import cube_to_fits
from astropy.io import fits
import matplotlib.pyplot as plt
from gofish import imagecube
from scipy.integrate import trapz, cumtrapz
from scipy import interpolate


# set velocities
velax = np.arange(-4000, 4000, 200.)
#velax = np.linspace(0, 1200, 6)

# load configs
inpr = importlib.import_module('gen_sg_sharpc')

# calculate cubes
fixedr = inpr.nu_rest, inpr.FOV[0], inpr.Npix[0], inpr.dist, inpr.cfg_dict


cuber = pardisk_radmc(velax, inpr.pars, fixedr)



# create FITS
cube_to_fits(cuber, 'cube_radmc.fits', RA=240., DEC=-30.)


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

# return the size
print('\n\n RADMC: ')
print("CO integrated flux = %f Jy km/s" % fluxr)
print("CO effective radius = %f au" % (inpr.dist * sizer))
