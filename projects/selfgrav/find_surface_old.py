import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from disksurf import observation
from eddy import linecube
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# user-specified model
mdl = 'sharpchi'
lbl = '$\mathsf{sharp \,\, c}$'



# style setups
import matplotlib.pyplot as plt
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


# Plot configuration
fig, ax = plt.subplots(figsize=(3.5, 2.25))
left, right, bottom, top = 0.14, 0.86, 0.17, 0.99

# axes limits
xlims = [0, 300]
ylims = [0, 105]

# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 500.5, 501), np.linspace(0.5, 100.5, 301)
z_ = np.logspace(-1.5, 2, 501)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)

# mark the model CO abundance region (top boundary is ~ emission surface)
Xco = abundance(rr, zz, inp, selfgrav=True)
ax.contour(r_, z_, Xco, levels=[inp.xmol], colors='k')

# disksurf extraction from raw cube
raw_cube = observation(inp.radmcname+'raw_cube.fits', 
                       FOV=xlims[1] * 2.5 / inp.dist, 
                       velocity_range=[-4000, 4000])

raw_surface = raw_cube.get_emission_surface(x0=0, y0=0, inc=inp.incl, 
                                            PA=inp.PA, vlsr=inp.Vsys)
raw_surface.mask_surface(side='front', min_zr=0.1, max_zr=0.35)

ax.scatter(inp.dist * raw_surface.r(side='front', masked=True), 
           inp.dist * raw_surface.z(side='front', masked=True), 
           color='gray', marker='.', alpha=0.1)

# tapered power-law fit 
p_, ep_ = raw_surface.fit_emission_surface(tapered_powerlaw=True, side='front', 
                                           masked=True, return_model=False)
print(p_)
zCO = inp.dist * p_[0] * (r_ / inp.dist)**p_[1] * \
      np.exp(-(r_ / (inp.dist * p_[2]))**p_[3])
ax.plot(r_, zCO, '--r', lw=1.5)


# limits
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# labels
ax.set_xlabel('$r$  (au)')
ax.set_ylabel('$z$  (au)')


# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

# save figure to output
fig.savefig('figs/'+mdl+'_COsurface.pdf')
fig.clf()





### EXTRACT v_phi(r) profile along this surface!

# Plot configuration
fig, ax = plt.subplots(nrows=2, figsize=(3.5, 4.2))
left, right, bottom, top, hsp = 0.14, 0.86, 0.09, 0.99, 0.30

# Keplerian velocity field along that surface
v_kep = omega_kep(r_ * sc.au, zCO * sc.au, inp) * r_ * sc.au
ax[0].plot(r_, 1e-3 * v_kep, ':', color='darkslategray')

# full velocity field on the grid
om2 = omega_kep(rr, zz, inp)**2 + \
      eps_P(rr, zz, inp, nselfgrav=True) + eps_g(rr, zz, inp)
v_grid = np.sqrt(om2) * rr

# interpolate in vertical direction to get the vphi(r) profile on surface
v_tot = np.empty_like(zCO)
for ir in range(len(zCO)):
    vzint = interp1d(z_, v_grid[:,ir], fill_value='extrapolate', kind='linear')
    v_tot[ir] = vzint(zCO[ir])
ax[0].plot(r_, 1e-3 * v_tot, '-k', lw=1.5)

# extract the vphi(r) from the raw cube
ercube = linecube(inp.radmcname+'raw_cube.fits', FOV=5.0)
ercube.data += np.random.normal(0, 1e-10, np.shape(ercube.data))

re, ve, dve = ercube.get_velocity_profile(x0=0, y0=0, inc=inp.incl,
                     PA=inp.PA, fit_vrad=True, fit_method='SNR',
                     get_vlos_kwargs=dict(centroid_method='doublegauss'),
                     rbins=np.arange(0.2, 2.0, 0.05), z0=p_[0], psi=p_[1],
                     r_taper=p_[2], q_taper=p_[3], r_cavity=0)
ax[0].errorbar(re * inp.dist, 
               1e-3 * ve[0] / np.sin(np.radians(inp.incl)), 
               1e-3 * dve[0] / np.sin(np.radians(inp.incl)), 
               fmt='.', color='r', ms=5, alpha=0.7)
                                         

# limits
ax[0].set_xlim(xlims)
ax[0].set_ylim([1, 20])
ax[0].set_yscale('log')

# labels
ax[0].set_yticks([1, 10])
ax[0].set_yticklabels(['1', '10'])
ax[0].set_xlabel('$r$  (au)')
ax[0].set_ylabel('$v_\\phi$  (km/s)', labelpad=7)


# plot the non-Keplerian residuals
ax[1].axhline(y=0, linestyle=':', color='darkslategray')
ax[1].plot(r_, v_tot - v_kep, '-k', lw=1.5)

vint = interp1d(r_, v_kep, fill_value='extrapolate', kind='linear')
ve_kep = vint(re * inp.dist)
ax[1].errorbar(re * inp.dist,
               ve[0] / np.sin(np.radians(inp.incl)) - ve_kep,
               dve[0] / np.sin(np.radians(inp.incl)), fmt='.', 
               color='C0', ms=5, alpha=0.7)


# limits
ax[1].set_xlim(xlims)
ax[1].set_ylim([-200, 200])

# labels
ax[1].set_xlabel('$r$  (au)')
ax[1].set_ylabel('$\delta v_{\phi}$  (m/s)', labelpad=-1)



# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hsp)

# save figure to output
fig.savefig('figs/'+mdl+'_vphi.pdf')
fig.clf()
