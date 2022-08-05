import os, sys, importlib
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import cmasher as cmr
sys.path.append('../../configs/')
sys.path.append('../../')
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from csalt.utils import *
from csalt.models import cube_to_fits, radmc_to_fits
from astropy.io import fits
import matplotlib.gridspec as gridspec
from matplotlib.colorbar import Colorbar
from matplotlib import mlab, cm
from astropy.visualization import (AsinhStretch, LinearStretch, ImageNormalize)
import cmasher as cmr
from scipy.interpolate import interp1d, griddata
from gofish import imagecube
from scipy.integrate import trapz, cumtrapz
from scipy import interpolate


# constants
_msun = 1.9891e33
_AU = sc.au * 1e2
_mu = 2.37  
_mH = (sc.m_e + sc.m_p) * 1e3
_k  = sc.k * 1e7
_G  = sc.G * 1e3

r_lims, zr_lims, z_lims = [0, 300], [0.0, 0.6], [0, 100]



# model setups
do_cube = True
cfg = 'radmc_std'
inp = importlib.import_module('gen_'+cfg)
fixed = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist, inp.cfg_dict


# cylindrical grid
r_ = np.logspace(-1, np.log10(300), 256)
z_ = np.logspace(-2, np.log10(300), 512)
rr, zz = np.meshgrid(r_, z_)

# scale height function
def Hp(r):
    cs = np.sqrt(_k * inp.Tmid0 * (r / 10)**inp.qmid / (_mu * _mH))
    om = np.sqrt(_G * inp.mstar * _msun / (r * _AU)**3)
    return cs / om

# CO surface height
zco = inp.zrmax * Hp(r_) / _AU


# T(r,z) in cylindrical space
def T_gas(r, z):
    r, z = np.atleast_1d(r), np.atleast_1d(z)
    Tmid, Tatm = inp.Tmid0 * (r / 10)**inp.qmid, inp.Tatm0 * (r / 10)**inp.qatm
    H = Hp(r) / _AU
    fz = 0.5 * np.tanh(((z/r) - inp.a_z * (H/r)) / (inp.w_z * (H/r))) + 0.5
    Tout = Tmid + fz * (Tatm - Tmid)
    return np.clip(Tout, a_min=0, a_max=1000)



# compute and plot T(r) along the CO emission surface
Tco = T_gas(r_, zco)
plt.style.use('default')
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(constrained_layout=True)
ax.plot(r_, Tco, '-k')
ax.plot(r_, inp.Tmid0 * (r_ / 10)**inp.qmid, '--C0')
ax.plot(r_, inp.Tatm0 * (r_ / 10)**inp.qatm, '--C1')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.3, 333])
ax.set_ylim([1, 1000])
ax.set_xlabel('$r$  (AU)')
ax.set_ylabel('$T_{\\rm CO}$  (K)')
fig.savefig('figs/Tco.png')
fig.clf()



# Plot T(r, z) in cylindrical space
Tcyl = T_gas(rr, zz)
plt.style.use('default')
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(constrained_layout=True)
im = ax.contourf(r_, z_, Tcyl, levels=np.linspace(0, 100, 21), cmap='plasma')
ax.contour(r_, z_, Tcyl, levels=[20])
ax.plot(r_, zco, '-k')
ax.set_xlim(r_lims)
ax.set_ylim(z_lims)
ax.set_xlabel('$r$  (AU)')
ax.set_ylabel('$z$  (AU)')
fig.colorbar(im, ax=ax, label='$T_{\\rm gas}$  (K)')
fig.savefig('figs/Tcyl.png')
fig.clf()

fig, ax = plt.subplots(constrained_layout=True)
ir0, ir1 = np.abs(r_ - 50).argmin(), np.abs(r_ - 150).argmin()
ax.plot(z_, Tcyl[:,ir0], 'C0')
ax.plot(z_, Tcyl[:,ir1], 'C1')
ax.set_xlim([0, 60])
ax.set_ylim([0, 100])
ax.set_xlabel('$z$  (AU)')
ax.set_ylabel('$T_{\\rm gas}$  (K)')
fig.savefig('figs/Tcyl_cuts.png')
fig.clf()



if do_cube:

    # set velocities
    velax = np.arange(-10000 + inp.Vsys, 10000 + inp.Vsys, 500.)

    # compute a cube
    cube = pardisk_radmc(velax, inp.pars, fixed)

    # create a FITS file
    cube_to_fits(cube, 'cube_radmc.fits', RA=240., DEC=-40.)


if os.path.exists('cube_radmc.fits'):

    ### PLOT OF SUBSET OF REPRESENTATIVE CHANNEL MAPS
    # load image and header information
    hdu = fits.open('cube_radmc.fits')
    Ico, hd = np.squeeze(hdu[0].data), hdu[0].header
    hdu.close()

    # define coordinate grids
    dx = 3600 * hd['CDELT1'] * (np.arange(hd['NAXIS1']) - (hd['CRPIX1'] - 1))
    dy = 3600 * hd['CDELT2'] * (np.arange(hd['NAXIS2']) - (hd['CRPIX1'] - 1))
    ext = (np.max(dx), np.min(dx), np.min(dy), np.max(dy))
    bm = np.abs(np.diff(dx)[0] * np.diff(dy)[0]) * (np.pi / 180)**2 / 3600**2

    # display properties
    vmin, vmax = 0., 80.   # these are in Tb / K
    lm = cm.get_cmap('cmr.pride', 20)
    xlims = np.array([1.8, -1.8])

    # r_l ellipse (in midplane)
    r_l = inp.r_l / inp.dist
    inclr, PAr = np.radians(inp.pars[0]), np.radians(inp.pars[1])
    tt = np.linspace(-np.pi, np.pi, 91)
    xgi = r_l * np.cos(tt) * np.cos(inclr)
    ygi = r_l * np.sin(tt)

    # set up plot (using 6 channels)
    fig = plt.figure(figsize=(7.5, 1.25))
    gs  = gridspec.GridSpec(1, 6, left=0.07, right=0.915, bottom=0.13, 
                            top=0.99, hspace=0., wspace=0.)

    for i in range(6):
        # convert intensities to brightness temperatures
        nu = hd['CRVAL3'] + (i + 19) * hd['CDELT3']
        Tb = (1e-26 * np.squeeze(Ico[i+19,:,:]) / bm) * sc.c**2 / \
             (2 * sc.k * nu**2)

        # plot the channel maps
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(Tb, origin='lower', cmap=lm, extent=ext, aspect='equal',
                       vmin=vmin, vmax=vmax)
        ax.plot( xgi * np.cos(PAr) + ygi * np.sin(PAr),
                -xgi * np.sin(PAr) + ygi * np.cos(PAr), ':w', lw=0.8)

        # limits / labeling
        ax.set_xlim(xlims)
        ax.set_ylim(-xlims)
        if (i == 0):
            ax.set_xlabel('$\Delta \\alpha$ ($^{\prime\prime}$)', labelpad=2)
            ax.set_ylabel('$\Delta \delta$ ($^{\prime\prime}$)', labelpad=-3)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    # colorbar
    cbax = fig.add_axes([0.92, 0.13, 0.012, 0.99-0.13])
    cb = Colorbar(ax=cbax, mappable=im, orientation='vertical',
                  ticklocation='right')
    cb.set_label('$T_b$  (K)', rotation=270, labelpad=15)

    fig.savefig('figs/vet_chmaps.png')



    ### COMPUTE CO FLUX AND SIZE
    # generate and load 0th moment map (made with bettermoments)
    os.system('bettermoments cube_radmc.fits -method zeroth -clip 0')
    cuber = imagecube('cube_radmc_M0.fits')
    bmr = np.abs(np.diff(dx)[0] * np.diff(dy)[0])

    # extract radial profile 
    xr, yr, dyr = cuber.radial_profile(inc=inp.pars[0], PA=inp.pars[1],
                                       x0=0.0, y0=0.0, PA_min=90, PA_max=270,
                                       abs_PA=True, exclude_PA=False)

    # convert to brightness units of Jy * km/s / arcsec**2
    yr /= 1000       # Jy m/s / pixel to Jy km/s / pixel
    yr /= bmr


    # integrated flux profile
    def fraction_curve(radius, intensity):
        intensity[np.isnan(intensity)] = 0
        total = trapz(2 * np.pi * radius * intensity, radius)
        cum = cumtrapz(2 * np.pi * radius * intensity, radius)
        return cum/total, total

    # size interpolator
    def Reff_fraction_smooth(radius, intensity, fraction=0.95):
        curve, flux = fraction_curve(radius, intensity)
        curve_smooth = interpolate.interp1d(curve, radius[1:])
        return curve_smooth(fraction), flux

    # get the size and the flux
    sizer, fluxr = Reff_fraction_smooth(xr, yr * np.cos(inclr), fraction=0.9)

    # return the size
    print('\n\n RADMC: ')
    print("CO integrated flux = %f Jy km/s" % fluxr)
    print("CO effective radius = %f au" % (inp.dist * sizer))


    # Load data from Long et al. 2021
    Jup, Fco, Rco, eRco, d, Mstar = np.loadtxt('data/feng_co.txt',
                                               usecols=(1,2,3,4,5,6), 
                                               skiprows=1).T

    # Proper adjustments for fluxes
    F_co = Fco
    F_co[Jup == 3] *= (230.538 / 345.796)**2

    # Set up plots
    fig, axs = plt.subplots(nrows=2, figsize=(3.5, 3.8), 
                            constrained_layout=True)

    # Mstar versus Fco
    ax = axs[0]
    ax.plot(Mstar, F_co * (d / 150)**2, marker='o', color='C0', mfc='C0',
            linestyle='None', ms=3.0, fillstyle='full')
    ax.plot([inp.mstar], [fluxr], marker='*', ms=9.5, color='C1', mfc='C1', 
            linestyle='None', fillstyle='full')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.08, 3.])
    ax.set_ylim([0.05, 200])
    ax.set_xticks([0.1, 1])
    ax.set_xticklabels([])
    ax.set_yticks([0.1, 1, 10, 100])
    ax.set_yticklabels(['0.1', '1', '10', '100'])
    ax.set_ylabel('$F_{\\rm CO} \\times (d / 150)^2 \,\,\, '+   
                  '[{\\rm Jy \, km/s}]$', fontsize=10.5)

    # Mstar versus Rco
    ax = axs[1]
    ax.errorbar(Mstar, Rco, yerr=eRco, fmt='o', mec='C0', mfc='C0', 
                elinewidth=1.5, ms=3.0, ecolor='C0', zorder=0)
    ax.plot([inp.mstar], [inp.dist * sizer], marker='*', ms=9.5, color='C1', 
            mfc='C1', linestyle='None', fillstyle='full')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim([0.08, 3.])
    ax.set_ylim([8, 2000])
    ax.set_xticks([0.1, 1])
    ax.set_xticklabels(['0.1', '1'])
    ax.set_yticks([10, 100, 1000])
    ax.set_yticklabels(['10', '100', '1000'])
    ax.set_xlabel('$M_\\ast \,\,\, [{\\rm M}_\\odot]$', fontsize=10.5)
    ax.set_ylabel('$R_{\\rm CO} \,\,\, [{\\rm au}]$', fontsize=10.5)

    fig.savefig('figs/vet_justify.png')



