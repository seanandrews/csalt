import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from parametric_disk_CSALT import parametric_disk as pardisk_csalt
from csalt.models import cube_to_fits
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)
import cmasher as cmr


# set velocities
velax = 5000 + np.array([-500., 0.0, 500., 1000., 1500., 2000.])
tvel = ['+%.1f km/s' % (1e-3 * i) for i in velax]

# load configs
inpr = importlib.import_module('gen_radmc2_slm')
inpc = importlib.import_module('gen_simp2_slm')

inpr.pars[0] = 50
inpc.pars[0] = 50

# calculate cubes
fixedr = inpr.nu_rest, inpr.FOV[0], inpr.Npix[0], inpr.dist, inpr.cfg_dict
#cuber = pardisk_radmc(velax, inpr.pars, fixedr)
fixedc = inpc.nu_rest, inpc.FOV[0], inpc.Npix[0], inpc.dist, inpc.cfg_dict
cubec = pardisk_csalt(velax, inpc.pars, fixedc)


# create FITS
#cube_to_fits(cuber, 'cube_radmc.fits', RA=240., DEC=-40.)
cube_to_fits(cubec, 'cube_csalt.fits', RA=240., DEC=-40.)


# make a plot of the channel maps for direct comparison

# load the FITS cubes 
hdu_r = fits.open('cube_radmc.fits')
hdu_c = fits.open('cube_csalt.fits')
Ico_r, Ico_c = np.squeeze(hdu_r[0].data), np.squeeze(hdu_c[0].data)
hd_r, hd_c = hdu_r[0].header, hdu_c[0].header

# define coordinate grids
#dx = 3600 * hd_c['CDELT1'] * (np.arange(hd_c['NAXIS1']) - (hd_c['CRPIX1'] - 1))
#dy = 3600 * hd_c['CDELT2'] * (np.arange(hd_c['NAXIS2']) - (hd_c['CRPIX1'] - 1))
dx = cubec.ra
dy = cubec.dec
ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
bm = np.abs(np.diff(dx)[0] * np.diff(dy)[0]) * (np.pi / 180)**2 / 3600**2

# display properties
vmin, vmax = 0., 70.   # these are in Tb / K
cm = cm.get_cmap('cmr.chroma_r', 10)
xlims = np.array([1.5, -1.5])

# r_l ellipse (in midplane)
r_l = inpr.r_l / inpr.dist
inclr, PAr = np.radians(inpr.pars[0]), np.radians(inpr.pars[1])
tt = np.linspace(-np.pi, np.pi, 91)
xgi = r_l * np.cos(tt) * np.cos(inclr)
ygi = r_l * np.sin(tt)

# set up plot (using 6 channels)
fig = plt.figure(figsize=(7.5, 3.0))
gs  = gridspec.GridSpec(2, 6)

for i in range(6):

    # convert intensities to brightness temperatures
    nu = hd_r['CRVAL3'] + i * hd_r['CDELT3']
    Tr = (1e-26 * np.squeeze(Ico_r[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)
    nu = hd_c['CRVAL3'] + i * hd_c['CDELT3']
    Tc = (1e-26 * np.squeeze(Ico_c[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)

    # plot the channel maps
    axr = fig.add_subplot(gs[0, i])
    axc = fig.add_subplot(gs[1, i])
    imr = axr.imshow(Tr, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    imc = axc.imshow(Tc, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)

    axr.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axc.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)

    if i == 0:
        axr.text(0.05, 0.93, 'RADMC-3D', ha='left', transform=axr.transAxes, 
                 fontsize=6, va='top')
        axc.text(0.05, 0.93, 'CSALT', ha='left', transform=axc.transAxes, 
                 fontsize=6, va='top')
    axr.text(0.95, 0.06, tvel[i], ha='right', transform=axr.transAxes,
             fontsize=5, color='darkslategrey')
    axc.text(0.95, 0.06, tvel[i], ha='right', transform=axc.transAxes,
             fontsize=5, color='darkslategrey')


    # modify the styling
    axr.set_xlim(xlims)
    axr.set_ylim(-xlims)
    axc.set_xlim(xlims)
    axc.set_ylim(-xlims)
    if (i == 0):
        axr.set_xticklabels([])
        axr.set_yticklabels([])
        axc.set_xlabel('$\Delta \\alpha$ ($^{\prime\prime}$)', labelpad=2)
        axc.text(-0.30, 0.5, '$\Delta \delta$ ($^{\prime\prime}$)',
                 horizontalalignment='center', verticalalignment='center',
                 rotation=90, transform=axc.transAxes)
    else:
        axr.set_xticklabels([])
        axr.set_yticklabels([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])


# colorbar
cbax = fig.add_axes([0.92, 0.138, 0.015, 0.807])
cb = Colorbar(ax=cbax, mappable=imc, orientation='vertical',
              ticklocation='right')
cb.set_label('brightness temperature (K)', rotation=270, labelpad=15)


fig.subplots_adjust(hspace=0.03, wspace=0.03)
fig.subplots_adjust(left=0.055, right=0.915, bottom=0.09, top=0.99)

fig.savefig('compare_RADMC_CSALT.png')
