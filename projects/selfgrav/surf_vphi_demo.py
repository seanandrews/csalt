import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.io import fits
from disksurf import observation
from eddy import linecube
import matplotlib.pyplot as plt



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


### SETUPS

# naming and labeling
mdl = 'sharpchi'
lbl = '$\mathsf{sharp \,\, c}$; default\nperfect ancillaries'


# axes limits
rlims = [0, 300]
zlims = [0, 105]
vlims = [1, 20]
dvlims = [-200, 200]

# which surface to use
fit_surf = False
msetup = '_exact'

# radial bins for velocity profile extraction
if mdl[-2:] == 'hi':
    beam = np.sqrt(0.117 * 0.100)
else:
    beam = np.sqrt(0.188 * 0.146)	
rbins_vphi = np.arange(0.05, 2.5, 0.25 * beam)


# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)

# set up plot grid
fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(7.5, 5.5))
left, right, bottom, top, hsp, wsp = 0.065, 0.935, 0.07, 0.99, 0.27, 0.33



### --- GET TRUTHS!

# load tau surface 3d locations
tau_locs = np.load(inp.radmcname+'/tausurface_3d.npz')['tau_locs']

# load raw cube
hdu = fits.open(inp.radmcname+'/raw_cube.fits')
Ico = np.squeeze(hdu[0].data)
hdu.close()

# find channels with peak intensity for each pixel
ix_peakI = np.argmax(Ico, axis=0)

# extract 3d surface locations at those channels for each pixel
tau_x, tau_y, tau_z = tau_locs[0,:,:,:], tau_locs[1,:,:,:], tau_locs[2,:,:,:]
xsurf, ysurf = np.empty(ix_peakI.shape), np.empty(ix_peakI.shape)
zsurf = np.empty(ix_peakI.shape)
for i in range(ix_peakI.shape[0]):
    for j in range(ix_peakI.shape[1]):
        xsurf[i, j] = tau_x[ix_peakI[i, j], i, j]
        ysurf[i, j] = tau_y[ix_peakI[i, j], i, j]
        zsurf[i, j] = tau_z[ix_peakI[i, j], i, j]
rsurf = np.sqrt(xsurf**2 + ysurf**2)


# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 500.5, 501), np.logspace(-1.5, 2, 501)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

# mark the model CO abundance region (top boundary is ~ emission surface)
Xco = abundance(rr, zz, inp, selfgrav=True)
    
# scatter plot of RADMC-3D defined tau = tau_s surface
cond0 = np.logical_and(zsurf > 0, ix_peakI > 0)
condd = np.logical_and(cond0, rsurf <= 280 * sc.au)
#ax.scatter(rsurf[condd] / sc.au, zsurf[condd] / sc.au, color='gray', 
#           marker='.', alpha=0.1)

### fit the surface to a tapered power-law model

# fit conditions / assignments
condf = np.logical_and(condd, 
                       ~np.logical_and(rsurf > 200*sc.au, zsurf/rsurf <= 0.27))
rf, zf = (rsurf[condf] / sc.au) / inp.dist, (zsurf[condf] / sc.au) / inp.dist
dz = 1. / (10 * np.ones_like(zf))

# fitted data cleanup / organization
nan_mask = np.isfinite(rf) & np.isfinite(zf) & np.isfinite(dz)
r, z, dz = rf[nan_mask], zf[nan_mask], dz[nan_mask]
idx = np.argsort(r)
r, z, dz = r[idx], z[idx], dz[idx]
dist = 1.
r0 = 1.

# fit setup and optimization
def _powerlaw(r, z0, q, r_cavity=0.0, r0=1.0):
    """Standard power law profile."""
    rr = np.clip(r, a_min=0.0, a_max=None)
    return z0 * (rr / r0)**q

def _tapered_powerlaw(r, z0, q, r_taper=np.inf, q_taper=1.0, r0=1.0):
    """Exponentially tapered power law profile."""
    rr = np.clip(r, a_min=0.0, a_max=None)
    f = _powerlaw(rr, z0, q, r0=r0)
    return f * np.exp(-(rr / r_taper)**q_taper)

kw = {} 
kw['maxfev'] = kw.pop('maxfev', 100000)
kw['sigma'] = dz

kw['p0'] = [0.3 * dist, 1.0, 1.0 * dist, 1.0]
def func(r, *args):
    return _tapered_powerlaw(r, *args, r0=r0)

try:
    popt, copt = curve_fit(func, r, z, **kw)
    copt = np.diag(copt)**0.5
except RuntimeError:
    popt = kw['p0']
    copt = [np.nan for _ in popt]

# overplot surface fit
zCO_true = inp.dist * popt[0] * (r_ / inp.dist)**popt[1] * \
           np.exp(-(r_ / (inp.dist * popt[2]))**popt[3])

# true velocity map on grid
om2 = omega_kep(rr, zz, inp)**2 + \
      eps_P(rr, zz, inp, nselfgrav=True) + \
      eps_g(rr, zz, inp)
v_grid = np.sqrt(om2) * rr

# interpolate in vertical direction to get the vphi(r) profile on surface
v_tot = np.empty_like(zCO_true)
for ir in range(len(zCO_true)):
    vzint = interp1d(z_, v_grid[:,ir], fill_value='extrapolate', kind='linear')
    v_tot[ir] = vzint(zCO_true[ir])
    

# keplerian velocity profile
v_kep = omega_kep(r_ * sc.au, zCO_true * sc.au, inp) * r_ * sc.au
vint = interp1d(r_, v_kep, fill_value='extrapolate', kind='linear')





### --- Configure loop over data complexities!
ipath = inp.reduced_dir+inp.basename+'/images/'
dfiles = [inp.radmcname+'/raw_cube.fits',
          ipath+inp.basename+'_pure.DATA.image.fits',
          ipath+inp.basename+'_noisy.DATA.image.fits']
dlbls = ['raw', 'sampled', 'noisy']

for i in range(len(dfiles)):

    ### SURFACE PLOTS
    ax = axs[0, i]

    # contour where gas-phase CO is abundant
    ax.contour(r_, z_, Xco, levels=[inp.xmol], colors='slategray')

    # best-fit surface profile from RADMC-3D optical depth maps
    ax.plot(r_, zCO_true, '-', color='darkslategray', lw=1.5)

    # extract surface fit from the emission cube
    cube = observation(dfiles[i], FOV=5.0,
                       velocity_range=[-4000, 4000])
    surface = cube.get_emission_surface(x0=0, y0=0, inc=inp.incl,
                                        PA=inp.PA, vlsr=inp.Vsys)
    surface.mask_surface(side='front', min_zr=0.05, max_zr=0.45)
    ax.scatter(inp.dist * surface.r(side='front', masked=True),
               inp.dist * surface.z(side='front', masked=True),
               color='gray', marker='.', alpha=0.1)
    if fit_surf:
        ps_, eps_ = surface.fit_emission_surface(tapered_powerlaw=True, 
                                                 side='front', masked=True,
                                                 return_model=False)
        zCO_meas = inp.dist * ps_[0] * (r_ / inp.dist)**ps_[1] * \
                   np.exp(-(r_ / (inp.dist * ps_[2]))**ps_[3])
        ax.plot(r_, zCO_meas, '--r', lw=1.5)

        psurf = 1. * np.array(ps_)
    else:
        psurf = 1. * np.array(popt)
        ax.plot(r_, zCO_true, '--r', lw=1.5)

    # limits and labeling
    ax.set_xlim(rlims)
    ax.set_ylim(zlims)
    if i == 0:
        ax.set_ylabel('$z$  (au)')
    ax.text(0.07, 0.93, dlbls[i], ha='left', va='top', transform=ax.transAxes)



    ### VPHI PROFILE PLOTS
    ax = axs[1, i]

    # true Keplerian profile
    v_kep = omega_kep(r_ * sc.au, zCO_true * sc.au, inp) * r_ * sc.au
    ax.plot(r_, 1e-3 * v_kep, ':', color='darkslategray')

    # true velocity profile
    ax.plot(r_, 1e-3 * v_tot, '-', color='darkslategray', lw=1.5)

    # extract the velocity profile from the cube!
    cube = linecube(dfiles[i], FOV=5.0)
    if i == 0:
        cube.data += np.random.normal(0, 1e-10, np.shape(cube.data))
    re, ve, dve = cube.get_velocity_profile(x0=0, y0=0, inc=inp.incl, 
                       PA=inp.PA, fit_vrad=True, fit_method='SNR',
                       get_vlos_kwargs=dict(centroid_method='gaussian'),
                       rbins=rbins_vphi, z0=psurf[0], psi=psurf[1], 
                       r_taper=psurf[2], q_taper=psurf[3], r_cavity=0)
    vphi_e = ve[0] / np.sin(np.radians(inp.incl))
    evphi_e = dve[0] / np.sin(np.radians(inp.incl))
    print(evphi_e)
    ax.errorbar(re * inp.dist, 1e-3 * vphi_e, 1e-3 * evphi_e, 
                fmt='.', color='r', ms=5)

    # limits and labeling
    ax.set_xlim(rlims)
    ax.set_ylim(vlims)
    ax.set_yscale('log')
    ax.set_yticks([1, 10])
    ax.set_yticklabels(['1', '10'])
    if i == 0:
        ax.set_ylabel('$v_\\phi$  (km/s)', labelpad=7)



    ### delta(VPHI) PROFILE PLOTS
    ax = axs[2, i]

    # true Keplerian profile
    ax.axhline(y=0, linestyle=':', color='darkslategray')

    # true non-Keplerian deviations profile
    ax.plot(r_, v_tot - v_kep, '-k', lw=1.5)

    # non-Keplerian contribution from cube (with perfect knowledge of v_kep)
    ax.errorbar(re * inp.dist, vphi_e - vint(re * inp.dist), evphi_e, 
                fmt='.', color='r', ms=5)

    # limits and labeling
    ax.set_xlim(rlims)
    ax.set_ylim(dvlims)
    if i == 0:
        ax.set_xlabel('$r$  (au)')
        ax.set_ylabel('$\delta v_{\phi}$  (m/s)', labelpad=-1)

# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                    hspace=hsp, wspace=wsp)



# save things into a npz file
np.savez('data/surf_vphi_'+mdl+msetup+'.npz', r_=r_, v_tot=v_tot, v_kep=v_kep,
         re=re*inp.dist, ve=vphi_e, eve=evphi_e, ve_kep=vint(re*inp.dist), 
         psurf=psurf)



# save figure to output
fig.savefig('figs/surf_vphi_'+mdl+msetup+'.pdf')
fig.clf()
