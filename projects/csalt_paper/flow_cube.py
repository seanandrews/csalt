import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from parametric_disk_CSALT import parametric_disk as pardisk_csalt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from matplotlib.colors import ListedColormap
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)
import cmasher as cmr


# set velocities
velax = 5000 + np.array([-6000, -500., 0.0, 500., 1000., 1500., 2000.])
velax = 5000 + np.arange(-6000, 6500, 500)


velax = 5000 + np.arange(-1500, 1650, 150)


# load configs
inpc = importlib.import_module('gen_simp2_slm')

# calculate cubes
fixedc = inpc.nu_rest, inpc.FOV[0], 4*inpc.Npix[0], inpc.dist, inpc.cfg_dict
cubec = pardisk_csalt(velax, inpc.pars, fixedc)


# load the FITS cubes 
Ico_c = np.rollaxis(np.fliplr(cubec.data), -1)

# define coordinate grids
dx, dy, freqs = cubec.ra, cubec.dec, cubec.freqs
ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
bm = np.abs(np.diff(dx)[0] * np.diff(dy)[0]) * (np.pi / 180)**2 / 3600**2

# display properties
vmin, vmax = 0., 70.   # these are in Tb / K
cmap = cm.get_cmap('cmr.ocean_r')
xlims = np.array([1.5, -1.5])

# set up plot
for i in range(len(velax)):

    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)

    # convert intensities to brightness temperatures
    nu = freqs[i]
    Tc = (1e-26 * np.squeeze(Ico_c[i,:,:]) / bm) * sc.c**2 / (2 * sc.k * nu**2)

    # generate an alpha map
    amap = np.zeros_like(Tc)
    amap[Tc > 0] = 1

    # plot the channel maps
    imc = ax.imshow(Tc, origin='lower', cmap=cmap, alpha=amap, extent=ext, 
                    aspect='equal', vmin=vmin, vmax=vmax)

    # modify the styling
    ax.set_xlim(xlims)
    ax.set_ylim(-xlims)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.savefig('figs/flow_cube_ch'+f"{i:02d}"+'.png', dpi=300)
