import os, sys, importlib
import numpy as np
import scipy.constants as sc
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])

# Load the desired font
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


# Load data
nu_T, nu_i, vis_i, nu_f, vis_f = np.loadtxt('setting_demo_data.txt').T


# Configure plot
fig, ax = plt.subplots(figsize=(3.5, 2.45))

xlims = [-2.5, 2.5]


# the "true" LSRK visibility spectrum
nu0 = 230.538e9 * (1 - 5000 / sc.c)
nu_ = np.concatenate((nu_i, nu_f))
vis_true = np.concatenate((vis_i, vis_f))[np.argsort(nu_)]
nu_true = np.sort(nu_)

# SRF convolution
chix = np.arange(500)
xch = chix - np.mean(chix)
SRF = 0.5 * np.sinc(xch) + 0.25 * np.sinc(xch - 40) + 0.25 * np.sinc(xch + 40)
vis_out = convolve1d(vis_true, SRF / np.sum(SRF), mode='nearest')

ax.plot(1e-6 * (nu_true - nu0), vis_true, '-', color='silver', zorder=0)
ax.plot(1e-6 * (nu_true[::40] - nu0), vis_out[::40], '.k', ms=5)
ax.set_xlim(xlims)
ax.set_xlabel('LSRK $\\nu - \\nu_{\\rm sys}$  (MHz)')
ax.set_ylabel('real  $\mathsf{V}_{\\nu}$  (Jy)', labelpad=2)

ax.text(0.025, 0.12, 'input (continuous)', transform=ax.transAxes,
        ha='left', color='darkgray', fontsize=8)
ax.text(0.025, 0.04, 'output (SRF-convolved)', 
        transform=ax.transAxes, ha='left', color='k', fontsize=8)


# secondary x-axis
def a2b(x):
    return -1e-6 * 230.538e9 * (x / sc.c)

def b2a(x):
    return -1e6 * (sc.c / x) / 230.538e9

secax = ax.secondary_xaxis('top', functions=(b2a, a2b))
secax.set_xlabel('LSRK $v - v_{\\rm sys}$  (m/s)', labelpad=7)

fig.subplots_adjust(bottom=0.170, top=0.815, left=0.14, right=0.86)
fig.savefig('figs/srf_demo.pdf')
fig.clf()
