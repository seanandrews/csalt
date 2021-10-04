import os, sys, time, importlib
import numpy as np
import scipy.constants as sc
from parametric_disk_chris import parametric_disk as pardisk_radmc
from csalt.models import cube_to_fits
sys.path.append('configs/')
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)

_AU = 1.496e13


# set velocities
velax = np.array([-500., 0.0, 500., 1000., 1500., 2000.])

# load configs
inpr = importlib.import_module('generate_chris')

# calculate structure
fixedr = inpr.nu_rest, inpr.FOV[0], inpr.Npix[0], inpr.dist, inpr.cfg_dict
foo = pardisk_radmc(velax, inpr.pars, fixedr)


# load the grid you've created (Rgrid = R, Tgrid = Theta)
_ = np.loadtxt(inpr.radmcname+'amr_grid.inp', skiprows=5, max_rows=1)
nr, nt = np.int(_[0]), np.int(_[1])
Rw = np.loadtxt(inpr.radmcname+'amr_grid.inp', skiprows=6, max_rows=nr+1)
Tw = np.loadtxt(inpr.radmcname+'amr_grid.inp', skiprows=nr+7, max_rows=nt+1)
Rgrid = 0.5*(Rw[:-1] + Rw[1:])
Tgrid = 0.5*(Tw[:-1] + Tw[1:])


# load the temperature structure T(Theta, R)
T = np.reshape(np.loadtxt(inpr.radmcname+'gas_temperature.inp', skiprows=2), 
               (nt, nr))


# plot the 2-d temperature structure
fig, ax = plt.subplots()
xx, yy = Rgrid / _AU, 0.5 * np.pi - Tgrid
levels = np.linspace(5, 200, 50)
contourf_kwargs = {}
cmap = contourf_kwargs.pop('cmap', 'plasma')
im = ax.contourf(xx, yy, T, levels=levels, cmap=cmap, **contourf_kwargs)
ax.set_ylim([0, np.pi/2])
plt.show()







#-----------

