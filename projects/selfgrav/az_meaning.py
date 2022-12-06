import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr

mdl = 'taper2hi'

# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)


# collect position angles of each pixel
cfile = inp.radmcname+'raw_cube.fits'
hdu = fits.open(cfile)
Ico, hd = np.squeeze(hdu[0].data), hdu[0].header
hdu.close()
ddx = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
ddy = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX1'] - 1))
ext = (np.max(ddx), np.min(ddx), np.min(ddy), np.max(ddy))
dx, dy = np.meshgrid(ddx, ddy)
inclr, PAr = np.radians(inp.incl), np.radians(inp.PA)
xd = (dx * np.cos(PAr) - dy * np.sin(PAr)) / np.cos(inclr)
yd = (dx * np.sin(PAr) + dy * np.cos(PAr))
azd = np.degrees(np.arctan2(yd, xd))
azs = np.degrees(np.arctan2(dy, dx))


ax = plt.subplot()
im = plt.imshow(azd, origin='lower', extent=ext, aspect='equal',
                vmin=-180, vmax=180, cmap=cmr.infinity)

# create an Axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax)

plt.show()
