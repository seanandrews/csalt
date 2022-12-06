import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from scipy.interpolate import interp1d
from astropy.io import fits
from eddy import linecube
from model_vexact import model_vphi
from model_vphi import model_vphi as model_simple
import matplotlib.pyplot as plt

from matplotlib import cm, font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


# user-specified model
mdl = 'taper2hi'
dtyp = 'noisy'
lbl = '$\mathsf{sharp \,\, c}$; default\nperfect ancillaries'

# additional controls
#

# plot configurations
xsize, ysize = 3.5, 4.0
left, right, bottom, top = 0.14, 0.86, 0.10, 0.99
hsp = 0.20
xlims = [0, 300]
vlims = [1, 20]
dvlims = [-200, 200]

# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)


# which surface to use
fit_surf = False

# radial bins for velocity profile extraction
if mdl[-2:] == 'hi':
    beam = np.sqrt(0.117 * 0.100)
else:
    beam = np.sqrt(0.188 * 0.146)	
rbins_vphi = np.arange(0.05, 1.9, 0.25 * beam)



# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 300.5, 301), np.logspace(-1.5, 2, 501)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

# mark the model CO abundance region (top boundary is ~ emission surface)
Xco = abundance(rr, zz, inp, selfgrav=True)
    

# true velocity map on grid
om_k = omega_kep(rr, zz, inp)
epsP_grid = eps_P(rr, zz, inp, nselfgrav=True)
epsg_grid = eps_g(rr, zz, inp)
omtot_grid = np.sqrt(om_k**2 + epsP_grid + epsg_grid)

# zCO surface
def zCO_func(r, pars, dpc=150):
    zco = dpc*pars[0] * (r/dpc)**pars[1] * np.exp(-(r/(dpc*pars[2]))**pars[3])
    return zco

# true surface
psurf = np.load('data/'+mdl+'.zCO_true.npz')['psurf']
zCO = zCO_func(r_, psurf, dpc=inp.dist)


# interpolate in vertical direction to get the vphi(r) profiles on surface
om_tot = np.empty_like(zCO)
epsp = np.empty_like(zCO)
epsg = np.empty_like(zCO)
for ir in range(len(zCO)):
    vtint = interp1d(z_, omtot_grid[:,ir], fill_value='extrapolate')
    om_tot[ir] = vtint(zCO[ir])

    vpint = interp1d(z_, epsP_grid[:,ir], fill_value='extrapolate')
    epsp[ir] = vpint(zCO[ir])

    vgint = interp1d(z_, epsg_grid[:,ir], fill_value='extrapolate')
    epsg[ir] = vgint(zCO[ir])


# profiles of interest
omkep_r = omega_kep(r_ * sc.au, zCO * sc.au, inp)
vkep_r = omkep_r * r_ * sc.au
vphi_r = om_tot * r_ * sc.au
dvphi_r = vphi_r - vkep_r
dvphip_r = (np.sqrt(omkep_r**2 + epsp) - omkep_r) * r_ * sc.au
dvphig_r = (np.sqrt(omkep_r**2 + epsg) - omkep_r) * r_ * sc.au


### Plot the full velocity field and non_keplerian deviations
# plot configuration
fig, axs = plt.subplots(figsize=(xsize, ysize), nrows=2)

## top plot: full vphi(r) field
ax = axs[0]

# true Keplerian profile
ax.plot(r_, 1e-3 * vkep_r, ':', color='dodgerblue', lw=1.5)

# true velocity profile
ax.plot(r_, 1e-3 * vphi_r, '-', color='k', lw=2.)

# extract the velocity profile from the cube!
if dtyp == 'raw':
    cube = linecube(inp.radmcname+'raw_cube.fits', FOV=5.0)
    cube.data += np.random.normal(0, 1e-10, np.shape(cube.data))
else:
    cube = linecube(inp.reduced_dir+inp.basename+'/images/'+\
                    inp.basename+'_'+dtyp+'.DATA.image.fits', FOV=5.0)
re, ve, dve = cube.get_velocity_profile(x0=0, y0=0, inc=inp.incl, 
                   PA=inp.PA, fit_vrad=True, fit_method='SNR',
                   get_vlos_kwargs=dict(centroid_method='Gaussian'),
                   rbins=rbins_vphi, z0=psurf[0], psi=psurf[1], 
                   r_taper=psurf[2], q_taper=psurf[3], r_cavity=0)
vphi_e = ve[0] / np.sin(np.radians(inp.incl))
evphi_e = dve[0] / np.sin(np.radians(inp.incl))
evphi_e = 10 * np.ones_like(vphi_e)
ax.errorbar(re * inp.dist, 1e-3 * vphi_e, 1e-3 * evphi_e, 
            fmt='.', color='gray', ms=5)


# models
t1 = time.time()
m1 = model_vphi(rbins_vphi * inp.dist, [1.0, 0.1])
t2 = time.time()
m2 = model_simple(rbins_vphi * inp.dist, [1.0, 0.1])
t3 = time.time()
print('hard model = %f s' % (t2 - t1))
print('easy model = %f s' % (t3 - t2))
#ax.plot(rbins_vphi * inp.dist, 1e-3 * m1, '-C1', lw=1.5, zorder=10)
#ax.plot(rbins_vphi * inp.dist, 1e-3 * m2, ':m', lw=1.5)

# limits and labeling
ax.set_xlim(xlims)
ax.set_ylim(vlims)
ax.set_yscale('log')
ax.set_yticks([1, 10])
ax.set_yticklabels(['1', '10'])
ax.set_ylabel('$v_\\phi$  (km/s)', labelpad=7)



### delta(VPHI) PROFILE PLOTS
ax = axs[1]

# true Keplerian profile
ax.axhline(y=0, linestyle=':', color='dodgerblue')

# non-Keplerian contributions
ax.plot(r_, dvphip_r, '--', color='C2', lw=1.5)
ax.plot(r_, dvphig_r, '--r', lw=1.5)

# true non-Keplerian deviations profile
ax.plot(r_, dvphi_r, '-k', lw=2)


# non-Keplerian contribution from cube (with perfect knowledge of v_kep)
vint = interp1d(r_, vkep_r, fill_value='extrapolate')
ax.errorbar(re * inp.dist, vphi_e - vint(re * inp.dist), evphi_e, 
            fmt='.', color='gray', ms=5)

# models
print(m1)
#ax.plot(rbins_vphi * inp.dist, m1 - vint(rbins_vphi * inp.dist), '-C1', lw=1.5, zorder=10)
#ax.plot(rbins_vphi * inp.dist, m2 - vint(rbins_vphi * inp.dist), ':m', lw=1.5)

# limits and labeling
ax.set_xlim(xlims)
ax.set_ylim(dvlims)
ax.set_xlabel('$r$  (au)')
ax.set_ylabel('$\delta v_{\phi}$  (m/s)', labelpad=-1)

# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hsp)



# save things into a npz file
np.savez('data/surf_vphi_'+mdl+'_'+dtyp+'.npz', 
         r_=r_, v_tot=vphi_r, v_kep=vkep_r,
         re=re*inp.dist, ve=vphi_e, eve=evphi_e, ve_kep=vint(re*inp.dist), 
         psurf=psurf)



# save figure to output
fig.savefig('figs/'+mdl+'_'+dtyp+'_vphi_wpts.pdf')
fig.clf()
