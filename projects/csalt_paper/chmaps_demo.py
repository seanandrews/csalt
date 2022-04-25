import os, sys, importlib
import numpy as np
import scipy.constants as sc
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
import cmasher as cmr
from matplotlib import font_manager

plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])

# Load the desired font
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


nchans = 7
vra = [[3640, 5560], [3960, 4920], [4120, 4600]]


# set up the figure
fig = plt.figure(figsize=(7.5, 3.5))
gs = gridspec.GridSpec(3, nchans, left=0.06, right=0.88, bottom=0.10, top=0.99,
                       hspace=0.30, wspace=0.03)


ctype = 'pure'

cdirs = ['../../storage/data/fiducial_snap/images/',
         '../../storage/data/fiducial_std/images/',
         '../../storage/data/fiducial_deep/images/']
cubes = ['fiducial_snap_'+ctype+'.DATA.image.fits',
         'fiducial_std_'+ctype+'.DATA.image.fits',
         'fiducial_deep_'+ctype+'.DATA.image.fits']
restfreq = 230.538e9

xlims = [2, -2]
ylims = [-2, 2]
vmin, vmax = 0, 80
cm = cmr.eclipse

for ia in range(3):

    # load the cube and header
    hdu = fits.open(cdirs[ia]+cubes[ia])
    Ico, hd = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # coordinate grids, beam parameters
    dx = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
    dy = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX2'] - 1))
    ext = (dx.max(), dx.min(), dy.min(), dy.max())
    bmaj, bmin, bpa = hd['BMAJ'], hd['BMIN'], hd['BPA']
    bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi / 180)**2

    # spectral information
    nu = hd['CRVAL3'] + hd['CDELT3'] * \
         (np.arange(hd['NAXIS3']) - (hd['CRPIX3'] - 1))
    vv = np.round(sc.c * (1 - nu / restfreq))

    # extract the channels of interest
    subset = np.logical_and(vv >= vra[ia][0], vv <= vra[ia][1])
    freqs, vels, cube = nu[subset], vv[subset], Ico[subset,:,:]

    # loop over channels
    for i in range(nchans):

        # convert intensities to brightness temperatures
        Tb = (1e-26 * np.squeeze(cube[i,:,:]) / bm) * sc.c**2 / \
             (2 * sc.k * freqs[i]**2)

        # allocate the panel
        ax = fig.add_subplot(gs[ia, i])
        pax = ax.get_position()

        # plot the channel map
        im = ax.imshow(Tb, origin='lower', cmap=cm, extent=ext, aspect='auto',
                       vmin=vmin, vmax=vmax)

        # set map boundaries
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # labeling
        if np.logical_and(ia == 2, i == 0):
            ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
            ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')
        else: 
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        

fig.savefig('figs/chmaps_'+ctype+'.pdf')
fig.clf()
