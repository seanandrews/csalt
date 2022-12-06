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
import cmasher as cmr
from matplotlib import cm, font_manager
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.style.use(['default', '/home/sandrews/mpl_styles/big_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


# model to plot
mdl = 'taper2hi'


# load input parameter dictionary
inp = importlib.import_module('gen_sg_'+mdl)


# cube file
cfile = inp.radmcname+'/raw_cube.fits'

# load the cube and header
hdu = fits.open(cfile)
Ico, hd = np.squeeze(hdu[0].data), hdu[0].header
hdu.close()

# define coordinate grids
dx = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
dy = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX1'] - 1))
ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
bm = np.abs(np.diff(dx)[0]*np.diff(dy)[0])*(np.pi/180)**2 / 3600**2

# display properties
vmin, vmax = 0, 70
nchans, ch0 = 6, 74
xlims = np.array([2.1, -2.1])
cm = cm.get_cmap('cmr.eclipse', 50)
norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())

for iv in range(Ico.shape[0]):

    # set up figure
    fig, ax = plt.subplots(figsize=(6.0, 5.46))

    # convert this channel map into brightness temperature
    nu = hd['CRVAL3'] + iv * hd['CDELT3']
    v = sc.c * (1 - nu / 230.538e9)
    Tb = (1e-26 * Ico[iv,:,:] / bm) * sc.c**2 / (2 * sc.k * nu**2)

    # plot the channel map
    im = ax.imshow(Tb, origin='lower', cmap=cm, extent=ext, aspect='equal',
                   norm=norm)

    # labels
    if np.abs(v) < 0.001: v = 0.0
    if np.logical_or(np.sign(v) == 1, np.sign(v) == 0):
        pref = '+'
    else:
        pref = ''
    vstr = pref+'%.2f km/s' % (1e-3 * v)
    print(vstr)
    ax.text(0.96, 0.94, vstr, transform=ax.transAxes, ha='right',
            va='center', fontsize=12, color='w')


    # modify the styling
    ax.set_xlim(xlims)
    ax.set_ylim(-xlims)
    ax.set_xticks([2, 1, 0, -1, -2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xlabel('$\Delta \\alpha$  ($^{\prime\prime}$)')
    ax.set_ylabel('$\Delta \\delta$  ($^{\prime\prime}$)', labelpad=2)

    # colorbar
    cbax = fig.add_axes([0.90, 0.10, 0.020, 0.88])
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                  ticklocation='right')
    cb.set_label('$T_{\\rm b}$  (K)', rotation=270, labelpad=12)


    fig.subplots_adjust(left=0.09, right=0.89, bottom=0.10, top=0.98)
    fig.savefig('movie/chanmap_'+str(iv).zfill(3)+'.jpg', dpi=200)
    fig.clf()

os.system('convert -delay 10 -loop 0 movie/*jpg chanmap_movie.gif')
