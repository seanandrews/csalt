import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from scipy.optimize import curve_fit
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import matplotlib.colors as mcolors
from matplotlib import mlab, cm
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
from astropy.visualization import (AsinhStretch, LogStretch, ImageNormalize)
import cmasher as cmr
from gofish import imagecube

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


# model to plot
mdl = 'taper2hi'


# load input parameter dictionary
inp = importlib.import_module('gen_sg_'+mdl)

# radial bins
rbins = np.arange(0, 2.5, 0.1)	#0.25 * 0.11)

# zCO params
psurf = None
psurf = [0.28530147, 1.45840787, 2.13575133, 5.09495478]

# load the Keplerian cube as the reference
rdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/'
kcube = imagecube(rdir+'sg_'+mdl+'_kep/raw_cube.fits', FOV=6.)

# residual colormap
c2 = plt.cm.Reds(np.linspace(0, 1, 32))
c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
c1 = np.vstack([c1, np.ones((12, 4))])
colors = np.vstack((c1, c2))
mymap = mcolors.LinearSegmentedColormap.from_list('eddymap', colors)
cmap = mymap


if psurf is None:

    # identify the Keplerian emission surface (presumed to not change!)
    tau_locs = np.load(rdir+'sg_'+mdl+'_kep/tausurface_3d.npz')['tau_locs']
    ix_peakI = np.argmax(kcube.data, axis=0)
    taux, tauy, tauz = tau_locs[0,:,:,:], tau_locs[1,:,:,:], tau_locs[2,:,:,:]
    xsurf, ysurf = np.empty(ix_peakI.shape), np.empty(ix_peakI.shape)
    zsurf = np.empty(ix_peakI.shape)
    for i in range(ix_peakI.shape[0]):
        for j in range(ix_peakI.shape[1]):
            xsurf[i, j] = taux[ix_peakI[i, j], i, j]
            ysurf[i, j] = tauy[ix_peakI[i, j], i, j]
            zsurf[i, j] = tauz[ix_peakI[i, j], i, j]
    rsurf = np.sqrt(xsurf**2 + ysurf**2)
    cond0 = np.logical_and(zsurf > 0, ix_peakI > 0)
    condd = np.logical_and(cond0, rsurf <= 280 * sc.au)
    condf = np.logical_and(condd,
                           ~np.logical_and(rsurf > 200*sc.au, 
                                           zsurf/rsurf <= 0.27))
    rf, zf = (rsurf[condf]/sc.au) / inp.dist, (zsurf[condf]/sc.au) / inp.dist
    dz = 1. / (10 * np.ones_like(zf))

    # fitted data cleanup / organization
    nan_mask = np.isfinite(rf) & np.isfinite(zf) & np.isfinite(dz)
    r, z, dz = rf[nan_mask], zf[nan_mask], dz[nan_mask]
    idx = np.argsort(r)
    r, z, dz = r[idx], z[idx], dz[idx]
    dist = 1.
    r0 = 1.

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

    psurf = 1. * np.array(popt)
    print(psurf)



# cube files
pdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_'+mdl+'_prs/'
adir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_'+mdl+'/'
sdir = '/pool/asha0/SCIENCE/csalt/storage/radmc/sg_'+mdl+'_sg/'
cfiles = [pdir+'raw_cube.fits',
          sdir+'raw_cube.fits',
          adir+'raw_cube.fits']
clbls = ['$\\varepsilon_P$', '$\\varepsilon_g$', 
         '$\\varepsilon_P + \\varepsilon_g$']


fig = plt.figure(figsize=(3.5, 6.0))
gs = gridspec.GridSpec(5, 1, height_ratios=(1, 0.65, 1, 1, 1))


ax = fig.add_subplot(gs[0, 0])
r, v, spec, rms = kcube.radial_spectra(rbins=rbins, x0=0, y0=0,
                        inc=inp.incl, PA=inp.PA, z0=psurf[0], psi=psurf[1],
                        r_cavity=None, r_taper=psurf[2], q_taper=psurf[3],
                        mstar=inp.mstar, dist=inp.dist, unit='K')

norm = ImageNormalize(vmin=0, vmax=50, stretch=AsinhStretch())
im = ax.pcolormesh(v / 1e3, r * inp.dist, spec, cmap='cmr.eclipse', norm=norm)

ax.text(0.03, 0.88, '$\Omega_{\\rm kep}$', color='w', transform=ax.transAxes,
        ha='left', va='center', fontsize=11)

ax.set_xlim([-2, 2])
ax.set_ylim([0, 300])


ax = fig.add_subplot(gs[1, 0])
ax.axis('off')

# colorbar
cbax = fig.add_axes([0.14, 0.925, 0.86-0.14, 0.01])
cb = Colorbar(ax=cbax, mappable=im, orientation='horizontal',
              ticklocation='top')
cb.set_label('$T_{\\rm b}$  (K)')




for ic in range(len(cfiles)):

    # load the cube and replace the data with the residual!
    rcube = imagecube(cfiles[ic], FOV=6.)
    rcube.data -= kcube.data

    # create the teardrop plot
    ax = fig.add_subplot(gs[ic+2, 0])
    r, v, spec, rms = rcube.radial_spectra(rbins=rbins, x0=0, y0=0, 
                            inc=inp.incl, PA=inp.PA, z0=psurf[0], psi=psurf[1], 
                            r_cavity=None, r_taper=psurf[2], q_taper=psurf[3], 
                            mstar=inp.mstar, dist=inp.dist, unit='K',
                            PA_min=-20, PA_max=20, abs_PA=True, 
                            exclude_PA=True, mask_frame='disk')

    im = ax.pcolormesh(v / 1e3, r * inp.dist, spec, cmap=cmap, 
                       vmin=-2.5, vmax=2.5)

    ax.text(0.03, 0.88, clbls[ic], color='k', transform=ax.transAxes,
            ha='left', va='center', fontsize=11)

    ax.set_xlim([-2, 2])
    ax.set_ylim([0, 300])
    if ic == 2:
        ax.set_xlabel('$v_{\\rm obs}$  (km/s)')
        ax.set_ylabel('$r$  (au)')
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

# residuals colorbar
cbax = fig.add_axes([0.14, 0.62, 0.86-0.14, 0.01])
cb = Colorbar(ax=cbax, mappable=im, orientation='horizontal',
              ticklocation='top')
cb.set_label('residual $T_{\\rm b}$  (K)')


fig.subplots_adjust(left=0.14, right=0.86, bottom=0.065, top=0.915, hspace=0.10)
fig.savefig('figs/teardrop_contribs_MAJOR.pdf')
