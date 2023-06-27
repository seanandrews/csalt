import os
import sys
import importlib
import numpy as np
import scipy.constants as sc
from parametric_disk_CSALT import parametric_disk as pd_csalt
from parametric_disk_RADMC3D_simpleX import parametric_disk as pd_radmc
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import cmasher as cmr
from astropy.io import fits
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
from csalt.helpers import *

# personal plot setups
_ = importlib.import_module('plot_setups')
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_img.mplstyle'])



# CSALT model parameters
inc, PA, mstar, vsys, dx, dy = 50, 90, 1.0, 0.0e3, 0., 0.
r_l, zf_0, zf_q, zb_0, zb_q = 250, 0.35, 1.25, 0.35, 1.25
Tf_0, Tf_q, Tb_0, Tb_q, Tm_0, Tm_q = 150, -0.5, 150, -0.5, 50, -0.5
tauf_0, tauf_q, taub_0, taub_q, taum_0, taum_q = 3.0, -1, 3.0, -1, 5.5, -1
pars_c = np.array([inc, PA, mstar, r_l, zf_0, zf_q, zb_0, zb_q, Tf_0, Tf_q,
                   Tb_0, Tb_q, Tm_0, Tm_q, 0., tauf_0, tauf_q, taub_0, taub_q,
                   taum_0, taum_q, vsys, dx, dy])

# RADMC model parameters
grid_params = {'spatial': {'nr': 360, 'nt': 240, 'r_min': 1.0, 'r_max': 500,
                           'rrefine': False, 'rref_i': [215], 'rref_o': [235],
                           'nrref': [50], 'rref_scl': ['lin']},
               'cyl': {'nr': 2048, 'nt': 2048, 'r_min': 0.1, 'r_max': 1000,
                       'z_min': 0.001, 'z_max': 500 }}

setup_params = {'incl_dust': 0, 'incl_lines': 1, 'nphot': 10000000,
                'scattering': 'Isotropic', 'camera_tracemode': 'image',
                'molecule': 'co', 'transition': 2, 'dustspec': 'DIANAstandard'}

cfg_dict = {'grid_params': grid_params, 'setup_params': setup_params,
            'isoz': False, 'dPdr': False, 'selfgrav': False,
            'dens_selfgrav': False,
            'radmcname': '/pool/asha0/SCIENCE/csalt/storage/radmc/test/'}

pars_r = np.array([inc, PA, mstar, r_l, Tm_0, Tf_0, Tm_q, Tf_q, 2., 3.,
                   50., -1.0, np.inf, 1e-5, 1e-3, 0.30, 0.36, 5000., 0.0, 
                   vsys, dx, dy])

# fixed parameters
p_fixed = 230.538e9, 5.13, 512, 150, cfg_dict

# velocity channels
velax = 1e3 * np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
nuax = 230.538e9 * (1 - velax / sc.c)

# compute the CSALT cube and save it to FITS 
c_cube = pd_csalt(velax, pars_c, p_fixed)
cube_to_fits(c_cube, 'csalt.fits', RA=240, DEC=-30)

# compute the RADMC-3D cube and save it to FITS
#r_cube = pd_radmc(velax, pars_r, p_fixed)
#cube_to_fits(r_cube, 'radmc.fits', RA=240, DEC=-30)

# load the cubes
hdu = fits.open('radmc.fits')
r_cube, rhd = hdu[0].data, hdu[0].header
hdu.close()
hdu = fits.open('csalt.fits')
c_cube, chd = hdu[0].data, hdu[0].header
hdu.close()


# plot the cubes
fig, axs = plt.subplots(nrows=3, ncols=len(velax), figsize=(15, 6.8))
fl, fr, fb, ft, hs, ws = 0.03, 0.93, 0.10, 0.99, 0.12, 0.03
xlims = [2., -2.]
ylims = [-2., 2.]

tmin, tmax = 5, 75
T_norm = ImageNormalize(vmin=tmin, vmax=tmax, stretch=LinearStretch())
T_cmap = cmr.chroma

rmin, rmax = -5, 5
r_norm = ImageNormalize(vmin=rmin, vmax=rmax, stretch=LinearStretch())
r_cmap = cmr.prinsenvlag_r

dx = 3600 * rhd['CDELT1'] * (np.arange(rhd['NAXIS1']) - (rhd['CRPIX1'] - 1))
dy = 3600 * rhd['CDELT2'] * (np.arange(rhd['NAXIS2']) - (rhd['CRPIX2'] - 1))
ext = (dx.max(), dx.min(), dy.min(), dy.max())
bm = np.abs(np.diff(dx)[0] * np.diff(dx)[0]) * ((np.pi / 180)**2 / 3600**2)


# loop over channels
for i in range(len(velax)):

    # Convert intensities to brightness temperatures
    T_radmc = (1e-26 * r_cube[i,:,:] / bm) * sc.c**2 / (2 * sc.k * nuax[i]**2)
    T_csalt = (1e-26 * c_cube[i,:,:] / bm) * sc.c**2 / (2 * sc.k * nuax[i]**2)

    # plot the RADMC-3D channel maps
    im_r = axs[0, i].imshow(T_radmc, origin='lower', cmap=T_cmap,
                            extent=ext, aspect='equal', norm=T_norm)

    # plot the CSALT channel maps
    im_c = axs[1, i].imshow(T_csalt, origin='lower', cmap=T_cmap,
                            extent=ext, aspect='equal', norm=T_norm)

    # plot the difference channel maps
    im_d = axs[2, i].imshow(T_radmc - T_csalt, origin='lower', cmap=r_cmap,
                            extent=ext, aspect='equal', norm=r_norm)


# Labels
axs[0,0].text(0.05, 0.87, 'RADMC-3D', color='w', transform=axs[0,0].transAxes)
axs[1,0].text(0.05, 0.87, 'CSALT', color='w', transform=axs[1,0].transAxes)
axs[2,0].text(0.05, 0.87, 'difference', color='k', transform=axs[2,0].transAxes)

# Map boundaries
plt.setp(axs, xlim=xlims, ylim=ylims)
plt.setp(axs, xticks=[], yticks=[], xticklabels=[], yticklabels=[])
axs[2, 0].set_xticks([2, 1, 0, -1, -2])
axs[2, 0].set_xticklabels(['2', '1', '0', '-1', '-2'])
axs[2, 0].set_yticks([-2, -1, 0, 1, 2])
axs[2, 0].set_yticklabels(['-2', '-1', '0', '1', '2'])
axs[2, 0].set_xlabel('RA offset  ($^{\prime\prime}$)')
axs[2, 0].set_ylabel('DEC offset  ($^{\prime\prime}$)')

# colorbar
pos = axs[1, -1].get_position()
cbax = fig.add_axes([fr+0.01, 0.408, 0.02, ft-0.408])
cb = Colorbar(ax=cbax, mappable=im_r, orientation='vertical',
              ticklocation='right')
cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=13)

pos = axs[2, -1].get_position()
cbax = fig.add_axes([fr+0.01, 0.10, 0.02, 0.274])
cb = Colorbar(ax=cbax, mappable=im_d, orientation='vertical',
              ticklocation='right')
cb.set_label('$\\Delta T_{\\rm b}$  (K)', rotation=270, labelpad=13)


fig.subplots_adjust(left=fl, right=fr, bottom=fb, top=ft, hspace=hs, wspace=ws)
fig.savefig('testdata/test_newprescription.pdf')
fig.clf()
