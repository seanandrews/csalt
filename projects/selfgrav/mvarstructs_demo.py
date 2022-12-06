import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from astropy.visualization import (SqrtStretch, LogStretch, ImageNormalize)

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
fig, axs = plt.subplots(nrows=5, ncols=3, figsize=(7.5, 8.25))
left, right, bottom, top = 0.06, 0.895, 0.05, 0.975
hspace, wspace = 0.40, 0.25

# axes configurations
xlims = [0, 300]
ylims = [0, 105.]

# colormaps
T_ra = [5, 140]
n_ra = [-0.5, 7.5]
v_ra = [-200, 200]
cmap_T = cm.get_cmap('cmr.sepia_r', 21)
cmap_n = cm.get_cmap('cmr.ocean_r', 21)
cmap_v = cm.get_cmap('RdBu_r', 35)


# configuration files
cfgs = ['gen_sg_taper2hi_M05', 'gen_sg_taper2hi', 'gen_sg_taper2hi_M15']
lbls = ['$M_{\\rm d} = 0.05$ $M_\\odot$', '$M_{\\rm d} = 0.10$ $M_\\odot$',
        '$M_{\\rm d} = 0.15$ $M_\\odot$']


# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 500.5, 501), np.linspace(0.5, 300.5, 301)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)
z__ = np.logspace(-1.5, 2, 501)
rr_, zz_ = np.meshgrid(r_ * sc.au, z__ * sc.au)
ext = (r_.min(), r_.max(), z_.min(), z_.max())


# Loop through models
for i in range(len(cfgs)):

    # Load configuration file as dictionary
    inp = importlib.import_module(cfgs[i])

    # compute the gas temperatures
    Tgas = temperature(rr, zz, inp)

    # plot the gas temperatures
    norm = ImageNormalize(vmin=T_ra[0], vmax=T_ra[1], stretch=SqrtStretch())
    imT = axs[0, i].imshow(Tgas, origin='lower', cmap=cmap_T, extent=ext, 
                           aspect='auto', norm=norm)

    # compute the gas densities
    ngas = numberdensity(rr, zz, inp, selfgrav=True)

    # plot the gas densities
    above = (ngas <= 100.)
    Xco = abundance(rr, zz, inp, selfgrav=True)
    imn = axs[1, i].imshow(np.log10(ngas * Xco),
                           origin='lower', cmap=cmap_n, extent=ext, 
                           aspect='auto', vmin=n_ra[0], vmax=n_ra[1])


    # compute the pressure support velocity residuals
    om_k = omega_kep(rr, zz, inp)
    om_prs2 = eps_P(rr, zz, inp, nselfgrav=True)
    om_tot = np.sqrt(om_k**2 + om_prs2)
    dvp = (om_tot - om_k) * rr
    dvp[above] = np.nan

    # plot the pressure support velocity residuals
    imvp = axs[2, i].imshow(dvp, origin='lower', cmap=cmap_v, extent=ext,
                            aspect='auto', vmin=v_ra[0], vmax=v_ra[1])

    # compute the self-gravity velocity residuals
    om_sg2 = eps_g(rr, zz, inp)
    om_tot = np.sqrt(om_k**2 + om_sg2)
    dvs = (om_tot - om_k) * rr
    dvs[above] = np.nan

    # plot the pressure support velocity residuals
    imvs = axs[3, i].imshow(dvs, origin='lower', cmap=cmap_v, extent=ext,
                            aspect='auto', vmin=v_ra[0], vmax=v_ra[1])

    # compute the total velocity residuals
    om_tot = np.sqrt(om_k**2 + om_prs2 + om_sg2)
    dv_ = (om_tot - om_k) * rr
    dv_[above] = np.nan

    # plot the total velocity residuals
    imv_ = axs[4, i].imshow(dv_, origin='lower', cmap=cmap_v, extent=ext,
                            aspect='auto', vmin=v_ra[0], vmax=v_ra[1])

    # show the CO emission surface
    Xco = abundance(rr_, zz_, inp, selfgrav=True)
    for ii in range(5):
        axs[ii, i].plot(r_, 0.1 * r_, ':', color='gray', lw=1)
        axs[ii, i].plot(r_, 0.2 * r_, ':', color='gray', lw=1)
        axs[ii, i].plot(r_, 0.3 * r_, ':', color='gray', lw=1)
        axs[ii, i].plot(r_, 0.4 * r_, ':', color='gray', lw=1)
        Hp_mid = H_pressure(r_ * sc.au, inp)
        axs[ii, i].plot(r_, Hp_mid / sc.au, '--', color='gray', lw=1.3)
        axs[ii, i].contour(r_, z__, Xco, levels=[inp.xmol], colors='k',
                           linewidths=1.)

    # limits
    [axs[j, i].set_xlim(xlims) for j in range(5)]
    [axs[j, i].set_ylim(ylims) for j in range(5)]

    # labels
    axs[0, i].text(0.5, 1.15, lbls[i], transform=axs[0, i].transAxes, color='k',
                ha='center', va='top', fontsize=10)
    [axs[j, 0].set_xlabel('$r$  (au)') for j in range(5)]
    [axs[j, 0].set_ylabel('$z$  (au)', labelpad=1) for j in range(5)]

# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                    hspace=hspace, wspace=wspace)

# colorbars
pos_T = axs[0, 2].get_position()
aspect_T = (pos_T.y1-pos_T.y0) / (pos_T.x1 - pos_T.x0)
cbax_T = fig.add_axes([right+0.015, pos_T.y0, 0.015, pos_T.y1-pos_T.y0])
cb_T = Colorbar(ax=cbax_T, mappable=imT, orientation='vertical', 
                ticklocation='right', ticks=[10, 20, 40, 60, 80, 100, 120, 140])
cb_T.set_label('$T$  (K)')

pos_n = axs[1, 2].get_position()
cbax_n = fig.add_axes([right+0.015, pos_n.y0, 0.015, pos_n.y1-pos_n.y0])
cb_n = Colorbar(ax=cbax_n, mappable=imn, orientation='vertical',
                ticklocation='right', ticks=[0, 1, 2, 3, 4, 5, 6])
cb_n.set_label('$\log_{10} \,\, n({\\rm gas})$  (cm$^{-3}$)')

pos_vp = axs[2, 2].get_position()
cbax_vp = fig.add_axes([right+0.015, pos_vp.y0, 0.015, pos_vp.y1-pos_vp.y0])
cb_vp = Colorbar(ax=cbax_vp, mappable=imvp, orientation='vertical',
                 ticklocation='right')
cb_vp.set_label('$\delta v_{\phi, P}$  (m/s)')

pos_vs = axs[3, 2].get_position()
cbax_vs = fig.add_axes([right+0.015, pos_vs.y0, 0.015, pos_vs.y1-pos_vs.y0])
cb_vs = Colorbar(ax=cbax_vs, mappable=imvs, orientation='vertical',
                 ticklocation='right')
cb_vs.set_label('$\delta v_{\phi, g}$  (m/s)')

pos_v_ = axs[4, 2].get_position()
cbax_v_ = fig.add_axes([right+0.015, pos_v_.y0, 0.015, pos_v_.y1-pos_v_.y0])
cb_v_ = Colorbar(ax=cbax_v_, mappable=imv_, orientation='vertical',
                 ticklocation='right')
cb_v_.set_label('$\delta v_{\phi}$  (m/s)')



# save figure to output
fig.savefig('figs/mvarstructs_demo.pdf')
fig.clf()
