import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *

sdir = 'storage/data/DECO_tests/'

# Read in the data MS
#ddict = read_MS(sdir+'Sz129_200ch.ms') 

# Instantiate a csalt model
cm = model('CSALT_DECO_taper_1_c')

# Define some fixed attributes for the modeling
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 160, 
            'Nup': 1, 'doppcorr': 'approx', 'online_avg': 2}

# Set the CSALT model parameters
pars = np.array([37.7, 149.76, 0.78, 136.46, 70.98, 
                 0.24, 1.38, 0.43, 2.08, 
                 85.01, -0.43, 50.73, -0.12, 
                 1.52, 4.01, 3.52, 0.8, 3.37, -2.92, 4294.8, 0.22, 0.39])

# Calculate a model dictionary; write it to model and residual MS files
#mdict = cm.modeldict(ddict, pars, kwargs=fixed_kw)
#write_MS(mdict, outfile=sdir+'Sz129_200ch_MODEL.ms')
#write_MS(mdict, outfile=sdir+'Sz129_200ch_RESID.ms', resid=True)


# Define some tclean keywords to be used in all imaging
tclean_kw = {'imsize': 256, 'start': '1.255km/s', 'width': '0.08km/s',
             'nchan': 77, 'restfreq': '230.538GHz', 'cell': '0.05arcsec',
             'scales': [0, 10, 30, 60], 'niter': 50000,
             'robust': 0.5, 'threshold': '2mJy'}

# Define some Keplerian mask keywords
kepmask_kw = {'inc': 37, 'PA': 150, 'mstar': 0.8, 'dist': 160, 'vlsr': 4255,
              'r_max': 1.7, 'nbeams': 1.5, 'zr': 0.24}

# Image the data, model, and residual cubes
#imagecube(sdir+'Sz129_200ch.ms', sdir+'Sz129_DATA', 
#          kepmask_kwargs=kepmask_kw, tclean_kwargs=tclean_kw)

tclean_kw['mask'] = sdir+'Sz129_DATA.mask'
#imagecube(sdir+'Sz129_200ch_MODEL.ms', sdir+'Sz129_MODEL', 
#          mk_kepmask=False, tclean_kwargs=tclean_kw)
#imagecube(sdir+'Sz129_200ch_RESID.ms', sdir+'Sz129_RESID',
#          mk_kepmask=False, tclean_kwargs=tclean_kw)



### Show the results!

cubes = [sdir+'Sz129_DATA', sdir+'Sz129_MODEL', sdir+'Sz129_RESID']
lbls = ['data', 'model', 'residual']

# Export the cubes to FITS format
#from casatasks import exportfits
#for i in range(len(cubes)):
#    exportfits(cubes[i]+'.image', fitsimage=cubes[i]+'.fits', 
#               velocity=True, overwrite=True) 


# Plot a subset of the cube as a direct comparison
import importlib
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse
from astropy.io import fits
import scipy.constants as sc
_ = importlib.import_module('plot_setups')
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_img.mplstyle'])
import cmasher as cmr

nchans = 7

fig, axs = plt.subplots(nrows=3, ncols=nchans, figsize=(7.5, 3.2))
fl, fr, fb, ft, hs, ws = 0.06, 0.88, 0.13, 0.99, 0.12, 0.03
xlims = [1.3, -1.3]
ylims = [-1.3, 1.3]

for ia in range(len(cubes)):

    # Load the cube and header data
    hdu = fits.open(cubes[ia]+'.fits')
    Ico, h = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # coordinate grids, beam parameters
    dx = 3600 * h['CDELT1'] * (np.arange(h['NAXIS1']) - (h['CRPIX1'] - 1))
    dx -= pars[-2]
    dy = 3600 * h['CDELT2'] * (np.arange(h['NAXIS2']) - (h['CRPIX2'] - 1))
    dy -= pars[-1]
    ext = (dx.max(), dx.min(), dy.min(), dy.max())
    bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
    bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi / 180)**2

    # spectral information
    vv = h['CRVAL3'] + h['CDELT3'] * (np.arange(h['NAXIS3']) - (h['CRPIX3']-1))
    ff = 230.538e9 * (1 - vv / sc.c)

    for i in range(nchans):

        # in-cube index
        j = 4 * i + 23

        # convert intensities to brightness temperatures
        Tb = (1e-26 * np.squeeze(Ico[j,:,:]) / bm) * sc.c**2 / \
             (2 * sc.k * ff[j]**2)

        # allocate the panel
        ax = axs[ia, i]
        pax = ax.get_position()

        # plot the channel map
        im = ax.imshow(Tb, origin='lower', cmap='cmr.cosmic', extent=ext,
                       aspect='auto', vmin=0, vmax=15)

        # set map boundaries
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

        # overlay beam dimensions
        beam = Ellipse((xlims[0] + 0.1*np.diff(xlims),
                        -xlims[0] - 0.1*np.diff(xlims)), 
                       3600 * bmaj, 3600 * bmin, angle=90-bpa)
        beam.set_facecolor('w')
        ax.add_artist(beam)

        # labeling
        if i == 0:
            ax.text(0.02, 0.90, lbls[ia], transform=ax.transAxes, ha='left',
                    va='center', style='italic', fontsize=8, color='w')
        if ia == 2:
            if np.abs(vv[j]) < 0.001: vv[j] = 0.0
            if np.logical_or(np.sign(vv[j]) == 1, np.sign(vv[j]) == 0):
                pref = '+'
            else:
                pref = ''
            vstr = pref+'%.2f' % (1e-3 * vv[j])
            ax.text(0.97, 0.08, vstr, transform=ax.transAxes, ha='right',
                    va='center', fontsize=7, color='w')
        if np.logical_and(ia == 2, i == 0):
            ax.set_xlabel('RA offset  ($^{\prime\prime}$)')
            ax.set_ylabel('DEC offset  ($^{\prime\prime}$)')
            ax.spines[['right', 'top']].set_visible(False)
        else:
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])

# colorbar
cbax = fig.add_axes([fr+0.01, fb+0.01, 0.02, ft-fb-0.02])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=13)

fig.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, hspace=hs, wspace=ws)
fig.savefig('demo_compare_DECO.pdf')
fig.clf()
