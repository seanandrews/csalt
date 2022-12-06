import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from matplotlib.patches import Ellipse
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)
import cmasher as cmr

# style setups
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
from matplotlib import cm, font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


# model to plot
mdl = 'taper2hi'


# load input parameter dictionary
inp = importlib.import_module('gen_sg_'+mdl)


# cube files
kdir = inp.reduced_dir+'sg_'+mdl+'_kep/images/'
pdir = inp.reduced_dir+'sg_'+mdl+'_prs/images/'
adir = inp.reduced_dir+'sg_'+mdl+'/images/'
cfiles = [kdir+'sg_'+mdl+'_kep_pure.DATA.image.fits',
          pdir+'sg_'+mdl+'_prs_pure.DATA.image.fits',
          adir+'sg_'+mdl+'_pure.DATA.image.fits']

rdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_'+mdl+'/'
adir = inp.reduced_dir+'sg_'+mdl+'/images/'
sdir = inp.reduced_dir+'sg_'+mdl+'_noSSP/images/'
cfiles = [rdir+'raw_cube.fits',
          rdir+'raw_cube_GaussPSF.fits',
          adir+'sg_'+mdl+'_pure.DATA.image.fits',
          sdir+'sg_'+mdl+'_noSSP_pure.DATA.image.fits']
clbls = ['$\\Omega_{\\rm kep}$', '$+ \\varepsilon_P$', '$+ \\varepsilon_g$',
         '$+ \\varepsilon_P + \\varepsilon_g$']
clbls = ['raw', 'Gaussian convolved', 'CLEAN', 'CLEAN no SSP']


# display properties
vmin, vmax = 0, 70
nchans, ch0, dch = 14, 74, 1
xlims = np.array([2.1, -2.1])
cm = cm.get_cmap('cmr.eclipse', 50)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

fig, axs = plt.subplots(nrows=4, ncols=nchans, figsize=(15, 4.4))#3.6))

for ic in range(len(cfiles)):

    # load the cube and header
    hdu = fits.open(cfiles[ic])
    Ico, hd = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    if ic == 0:
        Ico0 = 1. * Ico

    # define coordinate grids
    dx = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
    dy = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX1'] - 1))
    ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))

    # define beam areas (and dimensions for imaged cases)
    if ic <= 1:
        bm = np.abs(np.diff(dx)[0]*np.diff(dy)[0])*(np.pi/180)**2 / 3600**2
    #elif ic == 1:
    #    bm = (np.pi * 0.117 * 0.100 / (4*np.log(2))) * (np.pi/180)**2 / 3600**2
    else:
        bmaj, bmin, bpa = 3600 * hd['BMAJ'], 3600 * hd['BMIN'], hd['BPA']
        bm = (np.pi * bmaj * bmin / (4 * np.log(2))) * (np.pi/180)**2 / 3600**2

    # set up a row of channel maps
    for iv in range(nchans):

        # convert this channel map into brightness temperature
        nu = hd['CRVAL3'] + (dch*iv + ch0) * hd['CDELT3']
        v = sc.c * (1 - nu / 230.538e9)
        Tb = (1e-26*Ico[(dch*iv + ch0),:,:] / bm) * sc.c**2 / (2 * sc.k * nu**2)

        # plot the channel map
        ax = axs[ic, iv]
        im = ax.imshow(Tb, origin='lower', cmap=cm, extent=ext, aspect='auto',
                       norm=norm)

        # overlay the Keplerian channel map contours
        Tb0 = (1e-26*Ico0[(dch*iv+ch0),:,:] / bm) * sc.c**2 / (2 * sc.k * nu**2)
        #fT = (Tb - Tb0) 
#        ax.contour(dx, dy, fT, levels=[2, 4, 6], colors='r', linewidths=0.4)
#        ax.contour(dx, dy, fT, levels=[-6, -4, -2], colors='dodgerblue', 
#                   linewidths=0.4)

        # labels
        if iv == 0:
            ax.text(0.05, 0.9, clbls[ic], transform=ax.transAxes, ha='left',
                    va='center', style='italic', fontsize=8, color='w')
        if ic == 0:
            if np.abs(v) < 0.001: v = 0.0
            if np.logical_or(np.sign(v) == 1, np.sign(v) == 0):
                pref = '+'
            else: 
                pref = ''
            vstr = pref+'%.2f km/s' % (1e-3 * v)
            ax.text(0.97, 0.92, vstr, transform=ax.transAxes, ha='right',
                    va='center', fontsize=7, color='w')

        # beam dimensions
        if ic > 1:
            beam = Ellipse((xlims[0] + 0.1*np.diff(xlims),
                            -xlims[0] - 0.1*np.diff(xlims)), bmaj, bmin, 90-bpa)
            beam.set_facecolor('w')
            ax.add_artist(beam)

        # modify the styling
        ax.set_xlim(xlims)
        ax.set_ylim(-xlims)
        if np.logical_and(ic == 3, iv == 0):
            ax.set_yticks([-2, 0, 2])
            ax.set_xlabel('$\Delta \\alpha$  ($^{\prime\prime}$)')
            ax.set_ylabel('$\Delta \\delta$  ($^{\prime\prime}$)', labelpad=2)
        else:
            ax.tick_params(axis='both', which='both', length=0)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

# colorbar
cbax = fig.add_axes([0.963, 0.138, 0.006, 0.807])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
              ticklocation='right')
cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=13)


#fig.subplots_adjust(left=0.055, right=0.915, bottom=0.11, top=0.99,
#                    hspace=0.04, wspace=0.04)
fig.subplots_adjust(left=0.028, right=0.958, bottom=0.12, top=0.99,
                    hspace=0.04, wspace=0.04)
fig.savefig('figs/chanmap_vphir.pdf')
fig.clf()
