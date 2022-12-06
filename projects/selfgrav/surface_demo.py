import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from disksurf import observation
import matplotlib.pyplot as plt
from radmc_surf import radmc_surf

from matplotlib.colorbar import Colorbar
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
from matplotlib import cm, font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'



# user-specified model
mdl  = 'taper2hi'
dtyp = 'raw'
cmap = cm.get_cmap('Reds', 10)
cmap._segmentdata['alpha'][0] *= 0
#padded = ['white', *cm.Reds(np.arange(256))]
#cmap = mpl.colors.LinearSegmentedColormap.from_list('', padded)


# additional controls
get_ztruth = False
use_iter, Niter = False, 3

# plot configurations
xsize, ysize = 3.5, 2.2
left, right, bottom, top = 0.14, 0.835, 0.18, 0.98
xlims = [0, 300]
ylims = [0, 105]

# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)


### If desired, extract the true z_CO from the RADMC-3D tau = 2/3 surface
if np.logical_or(get_ztruth, ~os.path.exists('data/'+mdl+'.zCO_true.npz')):
    # get the "true" surface and a tapered power-law fit to it
    psurf, rsurf, zsurf = radmc_surf(inp.radmcname, dpc=inp.dist)

    # save this!
    np.savez('data/'+mdl+'.zCO_true.npz', 
             psurf=psurf, rsurf=rsurf, zsurf=zsurf)

else:
    d_surf = np.load('data/'+mdl+'.zCO_true.npz')
    psurf, rsurf, zsurf = d_surf['psurf'], d_surf['rsurf'], d_surf['zsurf']

# CO surface function
def zCO_func(r, pars, dpc=150):
    zco = dpc*pars[0] * (r/dpc)**pars[1] * np.exp(-(r/(dpc*pars[2]))**pars[3])
    return zco


### Extract surface from a cube
# filename for specified cube
if dtyp == 'raw':
    cfile = inp.radmcname+'raw_cube.fits'
else:
    cfile = inp.reduced_dir+inp.basename+'/images/'+inp.basename+'_' + \
            dtyp+'.DATA.image.fits'

# load the cube into disksurf format
cube = observation(cfile, FOV=6)

# get samples of the emission surface
surface = cube.get_emission_surface(x0=0, y0=0, inc=inp.incl, PA=inp.PA,
                                    vlsr=inp.Vsys)
surface.mask_surface(side='front', min_zr=0.05, max_zr=0.45)


if use_iter:
    # iterate on surface
    iter_surface = cube.get_emission_surface_iterative(surface, N=Niter)
    iter_surface.mask_surface(side='front', min_zr=0.05, max_zr=0.45)


### Plot the true and derived surfaces
# Plot configuration
fig, ax = plt.subplots(figsize=(xsize, ysize))

# 2-D grids (in meters)
r_, z_ = np.linspace(1, 301, 1201), np.linspace(0, 105, 630)#np.logspace(-1.5, 2, 501)
#r_, z_ = np.logspace(-1, 2.7, 256), np.logspace(-2, 2.2, 256)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

## abundance contour
Xco = abundance(rr, zz, inp, selfgrav=True)
ax.contour(r_, z_, Xco, levels=[inp.xmol], colors='k', linewidths=1.5, 
           linestyles='dotted', label='CO-rich layer')
ax.plot([0, 1], [0, 0], ':k', lw=1.5, label='CO-rich layer')

## model fit
ax.plot(r_, zCO_func(r_, psurf), '-c', lw=2, label='surface fit')

# histogram in 2-d of tau = 2/3 surface locations
redges, zedges = np.linspace(0, 300, 61), np.linspace(0, 150, 31)
zgrid, re, ze = np.histogram2d(rsurf.flatten()/sc.au, zsurf.flatten()/sc.au, 
                               bins=[redges, zedges], density=False)

# histogram of # of points in each r bin
rtot, re = np.histogram(rsurf.flatten()/sc.au, bins=redges, density=False)
rtotals = np.tile(rtot, (zgrid.shape[-1], 1))

# heat map of densities per r bin of the tau = 2/3 surface
ext = (redges.min(), redges.max(), zedges.min(), zedges.max())
im = ax.imshow(zgrid.T / rtotals, origin='lower', vmin=0, vmax=1,
               cmap=cmap, extent=ext, aspect='auto')

# limits
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# labels
ax.set_xlabel('$r$  (au)')
ax.set_ylabel('$z$  (au)')
#ax.text(0.05, 0.91, '$\\mathsf{taper 2}$ $\\tau_{\\rm los} = 2/3$', 
#        transform=ax.transAxes, ha='left', va='center', color='k', fontsize=8)

ax.legend(loc='upper left', prop={'size': 8}, 
          title='$\mathsf{taper 2}$: $\\tau_{\\rm los} = 2/3$', 
          title_fontsize=8)

# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


# colorbar
pos = ax.get_position()
cbax = fig.add_axes([right+0.015, pos.y0, 0.020, pos.y1-pos.y0])
cb = Colorbar(ax=cbax, mappable=im, orientation='vertical', 
              ticklocation='right')
cb.set_label('pixel fraction per $r$ bin', rotation=270, labelpad=13)

# save figure to output
fig.savefig('figs/surface_demo.pdf')
fig.clf()
