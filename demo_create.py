import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'

# Instantiate a csalt model
cm = model('CSALT')

# Create an empty MS from scratch
cdir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'
cm.template_MS('testdata/test.ms', 
               config=[cdir+'alma.cycle8.5.cfg', cdir+'alma.cycle8.8.cfg'],
               t_total=['2min', '7min'], t_integ='30s', observatory='ALMA',
               date=['2023/04/20', '2023/07/20'], HA_0=['0h', '1h'],
               restfreq=230.538e9, dnu_native=[122e3, 122e3],
               RA='16:00:00.00', DEC='-30:00:00.00')

# Get the data dictionary from the empty MS
ddict = read_MS('testdata/test.ms')

# Define some fixed attributes for the model
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 161,
            'Nup': 5, 'doppcorr': 'exact', 'noise_inject': 0.005}

# Set the CSALT model parameters
pars = np.array([-33, 90, 1.1, 120, 0.3, 1.5, 120, -0.5, 20., 217,
                 2.2, -1, 4100, 0, 0])

# Calculate a model dictionary; insert it to model MS files
pure_mdict, noisy_mdict = cm.modeldict(ddict, pars, kwargs=fixed_kw)
write_MS(pure_mdict, outfile='testdata/test_PURE.ms')
write_MS(noisy_mdict, outfile='testdata/test_NOISY.ms')


# Define some tclean keywords to be used in all imaging
tclean_kw = {'imsize': 1500, 'start': '2.35km/s', 'width': '0.35km/s',
             'nchan': 7, 'restfreq': '230.538GHz', 'cell': '0.01arcsec',
             'scales': [0, 10, 30, 60], 'niter': 50000,
             'robust': 1.0, 'threshold': '5mJy', 'uvtaper': '0.04arcsec'}

# Define some Keplerian mask keywords
kepmask_kw = {'inc': 33, 'PA': 150, 'mstar': 0.85, 'dist': 161, 'vlsr': 4100,
              'r_max': 1.1, 'nbeams': 1.5, 'zr': 0.2}

# Image the residual cubes
imagecube('testdata/test_PURE.ms', 'testdata/test_PURE',
          kepmask_kwargs=kepmask_kw, tclean_kwargs=tclean_kw)

tclean_kw['mask'] = 'testdata/test_PURE.mask'
imagecube('testdata/test_NOISY.ms', 'testdata/test_NOISY',
          mk_kepmask=False, tclean_kwargs=tclean_kw)




### Show the results!

cubes = ['testdata/test_PURE', 'testdata/test_NOISY']
lbls = ['pure', 'noisy']

# Export the cubes to FITS format
from casatasks import exportfits
for i in range(len(cubes)):
    exportfits(cubes[i]+'.image', fitsimage=cubes[i]+'.fits',
               velocity=True, overwrite=True)

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

fig, axs = plt.subplots(nrows=2, ncols=nchans, figsize=(7.5, 2.2))
fl, fr, fb, ft, hs, ws = 0.06, 0.88, 0.18, 0.99, 0.12, 0.03
xlims = [1.1, -1.1]
ylims = [-1.1, 1.1]

for ia in range(len(cubes)):

    # Load the cube and header data
    hdu = fits.open(cubes[ia]+'.fits')
    Ico, h = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # coordinate grids, beam parameters
    dx = 3600 * h['CDELT1'] * (np.arange(h['NAXIS1']) - (h['CRPIX1'] - 1))
    dy = 3600 * h['CDELT2'] * (np.arange(h['NAXIS2']) - (h['CRPIX2'] - 1))
    ext = (dx.max(), dx.min(), dy.min(), dy.max())
    bmaj, bmin, bpa = h['BMAJ'], h['BMIN'], h['BPA']
    bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi / 180)**2

    # spectral information
    vv = h['CRVAL3'] + h['CDELT3'] * (np.arange(h['NAXIS3']) - (h['CRPIX3']-1))
    ff = 230.538e9 * (1 - vv / sc.c)

    for i in range(nchans):

        # in-cube index
        j = i + 0

        # convert intensities to brightness temperatures
        Tb = (1e-26 * np.squeeze(Ico[j,:,:]) / bm) * sc.c**2 / \
             (2 * sc.k * ff[j]**2)

        # allocate the panel
        ax = axs[ia, i]
        pax = ax.get_position()

        # plot the channel map
        im = ax.imshow(Tb, origin='lower', cmap='cmr.cosmic', extent=ext,
                       aspect='auto', vmin=0, vmax=25)

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
        if ia == 1:
            if np.abs(vv[j]) < 0.001: vv[j] = 0.0
            if np.logical_or(np.sign(vv[j]) == 1, np.sign(vv[j]) == 0):
                pref = '+'
            else:
                pref = ''
            vstr = pref+'%.2f' % (1e-3 * vv[j])
            ax.text(0.97, 0.08, vstr, transform=ax.transAxes, ha='right',
                    va='center', fontsize=7, color='w')
        if np.logical_and(ia == 1, i == 0):
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
fig.savefig('testdata/demo_create.pdf')
fig.clf()





