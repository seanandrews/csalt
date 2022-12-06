import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.io import fits
from eddy import linecube
import matplotlib.pyplot as plt


# user-specified model
mdl = 'sharpc'
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



# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)

run_RT = False


# prepare for optical depth calculations
if run_RT:
    cwd = os.getcwd()
    os.chdir(inp.radmcname)
    os.system('mv image.out image0.out')
    if inp.incl < 0:
        incl = np.abs(inp.incl) + 180
        PA = inp.PA - 180
    else:
        incl = inp.incl
        PA = inp.PA
    posang = 90 - PA
    sizeau = inp.FOV[0] * inp.dist

    # raw cube channels
    velax = 1e3 * float(inp.chanstart[:-4]) + \
            1e3 * float(inp.chanwidth[:-4]) * np.arange(inp.nchan_out)
    wlax = 1e6 * sc.c / (inp.nu_rest * (1. - (velax - inp.Vsys) / sc.c))
    os.system('mv camera_wavelength_micron.inp camera_wavelength_micron0.inp')
    np.savetxt('camera_wavelength_micron.inp', wlax, 
               header=str(len(wlax))+'\n', comments='')

    # run RT for optical depth surface
    taus = 2./3.
    os.system('radmc3d tausurf %.2f ' % taus + \
              'incl %.2f ' % incl + \
              'posang %.2f ' % posang + \
              'npix %d ' % inp.Npix[0] + \
              'sizeau %d ' % sizeau + \
              'loadlambda ' + \
              'setthreads 6')
    
    # properly store outputs
    os.system('mv image.out tau_image.out')
    os.system('mv image0.out image.out')
    os.system('mv camera_wavelength_micron0.inp camera_wavelength_micron.inp')

    # load the tausurf positions into a proper cube array
    taufile = open('tausurface_3d.out')
    tformat = taufile.readline()
    im_nx, im_ny = taufile.readline().split()
    im_nx, im_ny = int(im_nx), int(im_ny)
    nlam = int(taufile.readline())

    tau_x, tau_y, tau_z = np.loadtxt('tausurface_3d.out', skiprows=4+nlam).T
    tau_x = np.reshape(tau_x, [nlam, im_ny, im_nx])
    tau_y = np.reshape(tau_y, [nlam, im_ny, im_nx])
    tau_z = np.reshape(tau_z, [nlam, im_ny, im_nx])

    # re-orient to align with the FITS output standard for a cube
    tau_x = np.rollaxis(np.fliplr(np.rollaxis(tau_x, 0, 3)), -1)
    tau_y = np.rollaxis(np.fliplr(np.rollaxis(tau_y, 0, 3)), -1)
    tau_z = np.rollaxis(np.fliplr(np.rollaxis(tau_z, 0, 3)), -1)

    # convert to a multi-dimensional array in SI (meters) units
    tau_locs = 1e-2 * np.stack((tau_x, tau_y, tau_z))

    # and save into a numpy binary file for easier access
    np.savez_compressed('tausurface_3d.npz', tau_locs=tau_locs)
    os.chdir(cwd)


# load tau surface 3d locations
tau_locs = np.load(inp.radmcname+'/tausurface_3d.npz')['tau_locs']

# load cube
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
    
# scatter plot of RADMC-3D defined tau = tau_s surface
cond0 = np.logical_and(zsurf > 0, ix_peakI > 0)
condd = np.logical_and(cond0, rsurf <= 280 * sc.au)
ax.scatter(rsurf[condd] / sc.au, zsurf[condd] / sc.au, color='gray', 
           marker='.', alpha=0.1)

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
zCO = inp.dist * popt[0] * (r_ / inp.dist)**popt[1] * \
      np.exp(-(r_ / (inp.dist * popt[2]))**popt[3])
ax.plot(r_, zCO, '--r', lw=1.5)

# overplot the disksurf surface fit
p_ = [0.34939652, 1.34755934, 3.38182778, 1.28560711]
zCO = inp.dist * p_[0] * (r_ / inp.dist)**p_[1] * \
      np.exp(-(r_ / (inp.dist * p_[2]))**p_[3])
ax.plot(r_, zCO, 'c', lw=1.5, zorder=1)

# limits
ax.set_xlim(xlims)
ax.set_ylim(ylims)

# labels
ax.set_xlabel('$r$  (au)')
ax.set_ylabel('$z$  (au)')


# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

# save figure to output
fig.savefig('figs/'+mdl+'_RADMCsurface.pdf')
fig.clf()




### EXTRACT v_phi(r) profile along these surfaces!

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
                     rbins=np.arange(0.2, 2.0, 0.1), z0=p_[0], psi=p_[1],
                     r_taper=p_[2], q_taper=p_[3], r_cavity=0)
ax[0].errorbar(re * inp.dist,
               1e-3 * ve[0] / np.sin(np.radians(inp.incl)),
               1e-3 * dve[0] / np.sin(np.radians(inp.incl)),
               fmt='.', color='c', ms=6, alpha=1.0)

rf, vf, dvf = ercube.get_velocity_profile(x0=0, y0=0, inc=inp.incl,
                     PA=inp.PA, fit_vrad=True, fit_method='SNR',
                     get_vlos_kwargs=dict(centroid_method='doublegauss'),
                     rbins=np.arange(0.2, 2.0, 0.1), z0=popt[0], psi=popt[1],
                     r_taper=popt[2], q_taper=popt[3], r_cavity=0)
ax[0].errorbar(rf * inp.dist,
               1e-3 * vf[0] / np.sin(np.radians(inp.incl)),
               1e-3 * dvf[0] / np.sin(np.radians(inp.incl)),
               fmt='.', color='r', ms=6, alpha=1.0)



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
               color='c', ms=6, alpha=1.0)
vf_kep = vint(rf * inp.dist)
ax[1].errorbar(rf * inp.dist,
               vf[0] / np.sin(np.radians(inp.incl)) - vf_kep,
               dvf[0] / np.sin(np.radians(inp.incl)), fmt='.',
               color='r', ms=6, alpha=1.0)


# limits
ax[1].set_xlim(xlims)
ax[1].set_ylim([-200, 200])

# labels
ax[1].set_xlabel('$r$  (au)')
ax[1].set_ylabel('$\delta v_{\phi}$  (m/s)', labelpad=-1)



# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top, hspace=hsp)

# save figure to output
fig.savefig('figs/'+mdl+'_RADMCvphi.pdf')
fig.clf()




