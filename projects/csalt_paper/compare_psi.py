import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
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
inpc = importlib.import_module('gen_fiducial_std')

# calculate cubes
fixedc = inpc.nu_rest, inpc.FOV[0], inpc.Npix[0], inpc.dist, inpc.cfg_dict
lpars = inpc.pars
lpars[5] = 0.5
print(lpars)
cubel = pardisk_csalt(velax, lpars, fixedc)
mpars = inpc.pars
mpars[5] = 1.0
print(mpars)
cubem = pardisk_csalt(velax, mpars, fixedc)
hpars = inpc.pars
hpars[5] = 1.5
print(hpars)
cubeh = pardisk_csalt(velax, hpars, fixedc)

# create FITS
cube_to_fits(cubel, 'lpsi.fits', RA=240., DEC=-40.)
cube_to_fits(cubem, 'mpsi.fits', RA=240., DEC=-40.)
cube_to_fits(cubeh, 'hpsi.fits', RA=240., DEC=-40.)



# make a plot of the channel maps for direct comparison

# load the FITS cubes 
hdu_l = fits.open('lpsi.fits')
hdu_m = fits.open('mpsi.fits')
hdu_h = fits.open('hpsi.fits')
Ico_l = np.squeeze(hdu_l[0].data)
Ico_m = np.squeeze(hdu_m[0].data)
Ico_h = np.squeeze(hdu_h[0].data)
hd_l, hd_m, hd_h = hdu_l[0].header, hdu_m[0].header, hdu_h[0].header
hdu_l.close()
hdu_m.close()
hdu_h.close()

# define coordinate grids
dx = 3600 * hd_l['CDELT1'] * (np.arange(hd_l['NAXIS1']) - (hd_l['CRPIX1'] - 1))
dy = 3600 * hd_l['CDELT2'] * (np.arange(hd_l['NAXIS2']) - (hd_l['CRPIX1'] - 1))
ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
bm = np.abs(np.diff(dx)[0] * np.diff(dy)[0]) * (np.pi / 180)**2 / 3600**2

# display properties
vmin, vmax = 0., 80.   # these are in Tb / K
cm = cm.get_cmap('cmr.chroma_r', 10)
xlims = np.array([1.8, -1.8])

# r_l ellipse (in midplane)
r_l = inpc.r_l / inpc.dist
inclr, PAr = np.radians(inpc.pars[0]), np.radians(inpc.pars[1])
tt = np.linspace(-np.pi, np.pi, 91)
xgi = r_l * np.cos(tt) * np.cos(inclr)
ygi = r_l * np.sin(tt)

# set up plot (using 6 channels)
fig = plt.figure(figsize=(7.5, 4.0))
gs  = gridspec.GridSpec(3, 6)

for i in range(6):

    # convert intensities to brightness temperatures
    nu = hd_l['CRVAL3'] + i * hd_l['CDELT3']
    Tl = (1e-26 * np.squeeze(Ico_l[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)
    nu = hd_m['CRVAL3'] + i * hd_m['CDELT3']
    Tm = (1e-26 * np.squeeze(Ico_m[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)
    nu = hd_h['CRVAL3'] + i * hd_h['CDELT3']
    Th = (1e-26 * np.squeeze(Ico_h[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)

    # plot the channel maps
    axl = fig.add_subplot(gs[0, i])
    axm = fig.add_subplot(gs[1, i])
    axh = fig.add_subplot(gs[2, i])
    iml = axl.imshow(Tl, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    imm = axm.imshow(Tm, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    imh = axh.imshow(Th, origin='lower', cmap=cm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)

    axl.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axm.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axh.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)

    if i == 0:
        axl.text(0.05, 0.93, '$\psi = 0.5$', ha='left', transform=axl.transAxes, 
                 fontsize=6, va='top')
        axm.text(0.05, 0.93, '$\psi = 1.0$', ha='left', transform=axm.transAxes, 
                 fontsize=6, va='top')
        axh.text(0.05, 0.93, '$\psi = 1.5$', ha='left', transform=axh.transAxes,
                 fontsize=6, va='top')
    axl.text(0.95, 0.06, tvel[i], ha='right', transform=axl.transAxes,
             fontsize=5, color='darkslategrey')
    axm.text(0.95, 0.06, tvel[i], ha='right', transform=axm.transAxes,
             fontsize=5, color='darkslategrey')
    axh.text(0.95, 0.06, tvel[i], ha='right', transform=axh.transAxes,
             fontsize=5, color='darkslategrey')


    # modify the styling
    axl.set_xlim(xlims)
    axl.set_ylim(-xlims)
    axm.set_xlim(xlims)
    axm.set_ylim(-xlims)
    axh.set_xlim(xlims)
    axh.set_ylim(-xlims)
    if (i == 0):
        axl.set_xticklabels([])
        axl.set_yticklabels([])
        axm.set_xticklabels([])
        axm.set_yticklabels([])
        axh.set_xlabel('$\Delta \\alpha$ ($^{\prime\prime}$)', labelpad=2)
        axh.text(-0.30, 0.5, '$\Delta \delta$ ($^{\prime\prime}$)',
                 horizontalalignment='center', verticalalignment='center',
                 rotation=90, transform=axh.transAxes)
    else:
        axl.set_xticklabels([])
        axl.set_yticklabels([])
        axm.set_xticklabels([])
        axm.set_yticklabels([])
        axh.set_xticklabels([])
        axh.set_yticklabels([])

# colorbar
cbax = fig.add_axes([0.92, 0.138, 0.015, 0.807])
cb = Colorbar(ax=cbax, mappable=imh, orientation='vertical',
              ticklocation='right')
cb.set_label('brightness temperature (K)', rotation=270, labelpad=15)


fig.subplots_adjust(hspace=0.03, wspace=0.03)
fig.subplots_adjust(left=0.055, right=0.915, bottom=0.09, top=0.99)

fig.savefig('compare_psi.png')
