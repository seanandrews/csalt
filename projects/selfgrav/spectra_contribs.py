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


# cube files
kepdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_taper2hi_kep/'
prsdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_taper2hi_prs/'
sgdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_taper2hi_sg/'
alldir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_taper2hi/'
cfiles = [kepdir+'raw_cube.fits',
          prsdir+'raw_cube.fits',
          sgdir+'raw_cube.fits',
          alldir+'raw_cube.fits']
clbls = ['$v = v_{\\ast}$', '$v =  v_{\\ast}{+}\delta v_P$', 
         '$v = v_{\\ast}{+}\delta v_g$', '$v = v_\\phi$']

ccols = ['C0', 'C2', 'C1', 'k']

left, right, bottom, top = 0.14, 0.86, 0.125, 0.99

xlims = [-4, 4]
ylims = [-0.1, 3.3]
yylims = [-0.5, 0.5]
yrat = np.diff(ylims)[0] / np.diff(yylims)[0]

# display properties
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 3.2),
                        gridspec_kw={'height_ratios': [yrat, 1]})

for ic in range(len(cfiles)):

    # load the cube and header
    hdu = fits.open(cfiles[ic])
    Ico, hd = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # get velocities
    nu = hd['CRVAL3'] + hd['CDELT3'] * np.arange(Ico.shape[0])
    v = sc.c * (1 - nu / 230.538e9)

    # get spectrum
    Fco = np.sum(Ico, axis=(1, 2))
    print(np.sum(Ico) * np.diff(v)[0])
    if ic == 0: Fco0 = 1.*Fco

    # plot the spectrum
    axs[0].plot(1e-3*v, Fco, color=ccols[ic], label=clbls[ic])

    # plot the residuals
    axs[1].plot(1e-3*v, Fco - Fco0, color=ccols[ic])

# legend
axs[0].legend(prop={'size': 6})

# modify the styling
axs[0].axhline(0, ls=':', color='gray')
axs[0].set_xlim(xlims)
axs[0].set_ylim(ylims)
axs[1].set_xlim(xlims)
axs[1].set_ylim(yylims)
axs[1].set_xlabel('$v_{\\rm obs} - v_{\\rm sys}$  (km/s)')
axs[0].set_ylabel('$F_v$  (Jy)', labelpad=7)
axs[1].set_ylabel('residual (Jy)', labelpad=3)

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=0.25)
fig.savefig('figs/spectra_contribs.pdf')
fig.clf()
