import os, sys, time
import numpy as np
from gofish import imagecube
from scipy.integrate import trapz, cumtrapz
from scipy import interpolate
from astropy.io import fits
import matplotlib.pyplot as plt

# model
mname = 'simple1-default'
fitscube = '../data/'+mname+'/'+mname+'_pure.DAT.image'

# model parameters
incl = 40.
PA = 130.
dpc = 150.


# generate and load 0th moment map (made with bettermoments)
os.system('bettermoments '+fitscube+'.fits -method zeroth -clip 0')
cube = imagecube(fitscube+'_M0.fits')
hd = fits.open(fitscube+'_M0.fits')[0].header
bmaj, bmin = 3600*hd['BMAJ'], 3600*hd['BMIN']

# extract radial profile 
x, y, dy = cube.radial_profile(inc=incl, PA=PA, x0=0.0, y0=0.0, 
                               PA_min=90, PA_max=270, 
                               abs_PA=True, exclude_PA=False)

# convert to brightness units of Jy * km/s / arcsec**2
y /= 1000	# Jy m/s / beam to Jy km/s / beam
y /= np.pi * bmaj * bmin / (4 * np.log(2))



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

    plt.plot(radius[1:], curve)
    plt.plot(curve_smooth(fraction), fraction, 'o')
    plt.show()

    return curve_smooth(fraction), flux


# get the size and the flux
size, flux = Reff_fraction_smooth(x, y, fraction=0.9)


# return the size
print(' ')
print("CO integrated flux = %f Jy km/s" % flux)
print("CO effective radius = %f au" % (dpc * size))
