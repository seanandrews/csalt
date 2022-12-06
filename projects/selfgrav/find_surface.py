import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from disksurf import observation
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from radmc_surf import radmc_surf

from matplotlib import font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'



# user-specified model
mdl  = 'taper2hi'
dtyp = 'pure'
lbl  = '$\mathsf{sharp \,\, c}$, '+dtyp

# additional controls
get_ztruth = False
use_iter, Niter = False, 3

# plot configurations
xsize, ysize = 3.5, 4.0
left, right, bottom, top = 0.14, 0.86, 0.10, 0.99
hspace = 0.36
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
fig, axs = plt.subplots(figsize=(xsize, ysize), nrows=2)

# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 300.5, 301), np.logspace(-1.5, 2, 501)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

# the model CO abundance region (top boundary is ~ emission surface)
Xco = abundance(rr, zz, inp, selfgrav=True)

## top plot: "truth"
ax = axs[0]
ax.contour(r_, z_, Xco, levels=[inp.xmol], colors='c')

# scatter plot of "true" surface measurements (tau = 2/3)
ax.scatter(rsurf / sc.au, zsurf / sc.au, 
           color='cornflowerblue', marker='.', alpha=0.1, rasterized=True)

# tapered power-law fit to the true surface measurements
ax.plot(r_, zCO_func(r_, psurf), '-b', lw=1.5)

# limits
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# labels
ax.set_ylabel('$z$  (au)')
ax.text(0.05, 0.91, 'truth: $\\tau = 2/3$ surface', transform=ax.transAxes,
        ha='left', va='center', color='b', fontsize=8)


## bottom plot: "measured"
ax = axs[1]
ax.contour(r_, z_, Xco, levels=[inp.xmol], colors='c')

# scatter plot of extracted surface measurements
if not use_iter:
    ax.scatter(inp.dist * surface.r(side='front', masked=True), 
               inp.dist * surface.z(side='front', masked=True), 
               color='pink', marker='.', alpha=0.1, rasterized=True)
    p_, ep_ = surface.fit_emission_surface(tapered_powerlaw=True,
                      side='front', masked=True, return_model=False)
else:
    ax.scatter(inp.dist * iter_surface.r(side='front', masked=True),
               inp.dist * iter_surface.z(side='front', masked=True),
               color='pink', marker='.', alpha=0.1, rasterized=True)
    p_, ep_ = iter_surface.fit_emission_surface(tapered_powerlaw=True, 
                           side='front', masked=True, return_model=False)

# plot tapered power-law fits
ax.plot(r_, zCO_func(r_, psurf), '-b', lw=1.5)
ax.plot(r_, zCO_func(r_, p_), '--r', lw=1.5)

print(psurf, p_)


# limits
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# labels
ax.set_xlabel('$r$  (au)')
ax.set_ylabel('$z$  (au)')
ax.text(0.05, 0.91, 'measured surface', transform=ax.transAxes,
        ha='left', va='center', color='r', fontsize=8)


# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

# save figure to output
fig.savefig('figs/'+mdl+'_'+dtyp+'_COsurface.pdf')
fig.clf()
