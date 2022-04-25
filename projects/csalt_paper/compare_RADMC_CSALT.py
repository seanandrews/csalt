import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from parametric_disk_CSALT import parametric_disk as pardisk_csalt
from csalt.models import cube_to_fits, radmc_to_fits
from csalt.utils import *
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
inpr = importlib.import_module('gen_radmc_std')
inpc = importlib.import_module('gen_fiducial_std')


# calculate cubes
fixedc = inpc.nu_rest, inpc.FOV[0], inpc.Npix[0], inpc.dist, inpc.cfg_dict
cubec = pardisk_csalt(velax, inpc.pars, fixedc)
fixedr = inpr.nu_rest, inpr.FOV[0], inpr.Npix[0], inpr.dist, inpr.cfg_dict
cuber = pardisk_radmc(velax, inpr.pars, fixedr)

# plot up some structural information for RADMC-3D model
r_lims, zr_lims = [0, 300], [0, 0.3]
_ = radmc_slice_plotter(inpr.radmcname, 'gas_temperature', xslice=[10, 100],
                        xlims=r_lims, ylims=zr_lims, zlims=[0,200],
                        lbl='$T_{\\rm gas}$  (K)')
_ = radmc_slice_plotter(inpr.radmcname, 'numberdens_co', xslice=[10, 100],
                        xlims=r_lims, ylims=zr_lims, zscl='log',
                        lbl='$n_{\\rm CO}$  (cm$^{-3}$)')


# create FITS
radmc_to_fits(inpr.radmcname+'/', 'cube_radmc.fits', fixedr)
cube_to_fits(cubec, 'cube_csalt.fits', RA=240., DEC=-40.)



# make a plot of the channel maps for direct comparison

# load the FITS cubes 
hdu_r = fits.open('cube_radmc.fits')
hdu_c = fits.open('cube_csalt.fits')
Ico_r, Ico_c = np.squeeze(hdu_r[0].data), np.squeeze(hdu_c[0].data)
hd_r, hd_c = hdu_r[0].header, hdu_c[0].header

# define coordinate grids
dx = 3600 * hd_c['CDELT1'] * (np.arange(hd_c['NAXIS1']) - (hd_c['CRPIX1'] - 1))
dy = 3600 * hd_c['CDELT2'] * (np.arange(hd_c['NAXIS2']) - (hd_c['CRPIX1'] - 1))
ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
bm = np.abs(np.diff(dx)[0] * np.diff(dy)[0]) * (np.pi / 180)**2 / 3600**2

# display properties
vmin, vmax = 0., 80.   # these are in Tb / K
lm = cm.get_cmap('cmr.chroma_r', 20)
dm = cm.get_cmap('cmr.prinsenvlag', 11)
xlims = np.array([1.5, -1.5])

# r_l ellipse (in midplane)
r_l = inpr.r_l / inpr.dist
inclr, PAr = np.radians(inpr.pars[0]), np.radians(inpr.pars[1])
tt = np.linspace(-np.pi, np.pi, 91)
xgi = r_l * np.cos(tt) * np.cos(inclr)
ygi = r_l * np.sin(tt)

# set up plot (using 6 channels)
fig = plt.figure(figsize=(7.5, 3.75))
gs  = gridspec.GridSpec(3, 6)

for i in range(6):

    # convert intensities to brightness temperatures
    nu = hd_r['CRVAL3'] + i * hd_r['CDELT3']
    Tr = (1e-26 * np.squeeze(Ico_r[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)
    nu = hd_c['CRVAL3'] + i * hd_c['CDELT3']
    Tc = (1e-26 * np.squeeze(Ico_c[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)

    # plot the channel maps
    axr = fig.add_subplot(gs[0, i])
    axc = fig.add_subplot(gs[1, i])
    posc = axc.get_position() 
    axd = fig.add_subplot(gs[2, i])
    posd = axd.get_position()

    imr = axr.imshow(Tr, origin='lower', cmap=lm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    imc = axc.imshow(Tc, origin='lower', cmap=lm, extent=ext, aspect='equal',
                     vmin=vmin, vmax=vmax)
    axc.contour(dx, dy, Tr, levels=[26.], color='k', linewidths=0.5)
    imd = axd.imshow(Tr - Tc, origin='lower', cmap=dm, extent=ext,
                     aspect='equal', vmin=-7, vmax=7)

    axr.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axc.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)
    axd.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
             -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':k', lw=0.8)

    if i == 0:
        axr.text(0.05, 0.93, 'RADMC-3D', ha='left', transform=axr.transAxes, 
                 fontsize=6, va='top')
        axc.text(0.05, 0.93, 'CSALT', ha='left', transform=axc.transAxes, 
                 fontsize=6, va='top')
        axd.text(0.05, 0.93, 'diff', ha='left', transform=axd.transAxes,
                 fontsize=6, va='top')
    axr.text(0.95, 0.06, tvel[i], ha='right', transform=axr.transAxes,
             fontsize=5, color='darkslategrey')
    axc.text(0.95, 0.06, tvel[i], ha='right', transform=axc.transAxes,
             fontsize=5, color='darkslategrey')
    axd.text(0.95, 0.06, tvel[i], ha='right', transform=axd.transAxes,
             fontsize=5, color='darkslategrey')


    # modify the styling
    axr.set_xlim(xlims)
    axr.set_ylim(-xlims)
    axc.set_xlim(xlims)
    axc.set_ylim(-xlims)
    axd.set_xlim(xlims)
    axd.set_ylim(-xlims)
    if (i == 0):
        axr.set_xticklabels([])
        axr.set_yticklabels([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axd.set_xlabel('$\Delta \\alpha$ ($^{\prime\prime}$)', labelpad=2)
        axd.set_ylabel('$\Delta \delta$ ($^{\prime\prime}$)', labelpad=-3)
    else:
        axr.set_xticklabels([])
        axr.set_yticklabels([])
        axc.set_xticklabels([])
        axc.set_yticklabels([])
        axd.set_xticklabels([])
        axd.set_yticklabels([])


# colorbar
cbax = fig.add_axes([0.92, posc.y0 + 0.05, 0.012, 0.93-posc.y0])
cb = Colorbar(ax=cbax, mappable=imc, orientation='vertical',
              ticklocation='right')
cb.set_label('$T_b$  (K)', rotation=270, labelpad=15)

cbax = fig.add_axes([0.92, posd.y0 + 0.04, 0.012, posc.y0-posd.y0 - 0.03])
cb = Colorbar(ax=cbax, mappable=imd, orientation='vertical',
              ticklocation='right')
cb.set_label('$\Delta T_b$  (K)', rotation=270, labelpad=12)


fig.subplots_adjust(hspace=0.0, wspace=0.0)
fig.subplots_adjust(left=0.07, right=0.915, bottom=0.13, top=0.99)

fig.savefig('compare_RADMC_CSALT.png')
fig.clf()
