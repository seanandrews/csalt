import os, sys, time, importlib
import numpy as np
import scipy.constants as sc
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from csalt.models import cube_to_fits
sys.path.append('configs/')
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)
import cmasher as cmr


# set velocities
velax = np.array([-500., 0.0, 500., 1000., 1500., 2000.])

# load configs
inpr = importlib.import_module('generate_phys2-default')
inpp = importlib.import_module('generate_phys2-pressure')
inpg = importlib.import_module('generate_phys2-selfgrav')

# calculate cubes
fixedr = inpr.nu_rest, inpr.FOV[0], inpr.Npix[0], inpr.dist, inpr.cfg_dict
#cuber = pardisk_radmc(velax, inpr.pars, fixedr)
fixedp = inpp.nu_rest, inpp.FOV[0], inpp.Npix[0], inpp.dist, inpp.cfg_dict
#cubep = pardisk_radmc(velax, inpp.pars, fixedp)
fixedg = inpg.nu_rest, inpg.FOV[0], inpg.Npix[0], inpg.dist, inpg.cfg_dict
#cubeg = pardisk_radmc(velax, inpg.pars, fixedg)


# create FITS
#cube_to_fits(cuber, 'cube_kepler.fits', RA=240., DEC=-40.)
#cube_to_fits(cubep, 'cube_pressure.fits', RA=240., DEC=-40.)
#cube_to_fits(cubeg, 'cube_selfgrav.fits', RA=240., DEC=-40.)


# make a plot of the channel maps for direct comparison

# load the FITS cubes 
hdu_r = fits.open('cube_kepler.fits')
Ico_r, hd_r = np.squeeze(hdu_r[0].data), hdu_r[0].header
hdu_p = fits.open('cube_pressure.fits')
Ico_p, hd_p = np.squeeze(hdu_p[0].data), hdu_p[0].header
hdu_g = fits.open('cube_selfgrav.fits')
Ico_g, hd_g = np.squeeze(hdu_g[0].data), hdu_g[0].header

# define coordinate grids
dx = 3600 * hd_r['CDELT1'] * (np.arange(hd_r['NAXIS1']) - (hd_r['CRPIX1'] - 1))
dy = 3600 * hd_r['CDELT2'] * (np.arange(hd_r['NAXIS2']) - (hd_r['CRPIX1'] - 1))
ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
bm = np.abs(np.diff(dx)[0] * np.diff(dy)[0]) * (np.pi / 180)**2 / 3600**2

# display properties
vmin, vmax = 0., 50.   # these are in Tb / K
cm = cm.get_cmap('cmr.chroma_r', 15)
xlims = np.array([2.5, -2.5])

# r_l ellipse (in midplane)
r_l, inclr, PAr = 150. / 150., np.radians(40.), np.radians(130.)
tt = np.linspace(-np.pi, np.pi, 91)
xgi = r_l * np.cos(tt) * np.cos(inclr)
ygi = r_l * np.sin(tt)


# set up plot 
fig = plt.figure(figsize=(7.75, 3.75))
gs  = gridspec.GridSpec(3, len(velax))

for i in range(len(velax)):

    # convert intensities to brightness temperatures
    nu = hd_r['CRVAL3'] + i * hd_r['CDELT3']
    Tr = (1e-26 * np.squeeze(Ico_r[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)
    nu = hd_p['CRVAL3'] + i * hd_p['CDELT3']
    Tp = (1e-26 * np.squeeze(Ico_p[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)
    nu = hd_g['CRVAL3'] + i * hd_g['CDELT3']
    Tg = (1e-26 * np.squeeze(Ico_g[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)

    # plot the channel maps
    axr = fig.add_subplot(gs[0, i])
    axp = fig.add_subplot(gs[1, i])
    axg = fig.add_subplot(gs[2, i])
    imr = axr.imshow(Tr, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    imp = axp.imshow(Tp, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    img = axg.imshow(Tg, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)

    axr.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axp.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axg.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)

    # modify the styling
    axr.set_xlim(xlims)
    axr.set_ylim(-xlims)
    axp.set_xlim(xlims)
    axp.set_ylim(-xlims)
    axg.set_xlim(xlims)
    axg.set_ylim(-xlims)
    axg.set_xticks([2, 0, -2])
    if (i == 0):
        axr.text(0.1, 0.1, 'rotation only', ha='left', fontsize=8, 
                 transform=axr.transAxes)
        axp.text(0.1, 0.1, '+ pressure', ha='left', fontsize=8,
                 transform=axp.transAxes)
        axg.text(0.1, 0.1, '+ self-gravity', ha='left', fontsize=8,
                 transform=axg.transAxes)
        axr.set_xticklabels([])
        axr.set_yticklabels([])
        axp.set_xticklabels([])
        axp.set_yticklabels([])
        axg.set_xlabel('$\Delta \\alpha$ ($^{\prime\prime}$)', labelpad=2)
        axg.text(-0.30, 0.5, '$\Delta \delta$ ($^{\prime\prime}$)',
                 horizontalalignment='center', verticalalignment='center',
                 rotation=90, transform=axg.transAxes)
    else:
        axr.set_xticklabels([])
        axr.set_yticklabels([])
        axp.set_xticklabels([])
        axp.set_yticklabels([])
        axg.set_xticklabels([])
        axg.set_yticklabels([])


# colorbar
cbax = fig.add_axes([0.92, 0.138, 0.015, 0.807])
cb = Colorbar(ax=cbax, mappable=imr, orientation='vertical',
              ticklocation='right')
cb.set_label('brightness temperature (K)', rotation=270, labelpad=15)


fig.subplots_adjust(hspace=0.03, wspace=0.03)
fig.subplots_adjust(left=0.055, right=0.915, bottom=0.11, top=0.99)

fig.savefig('test_RADMC.png')
