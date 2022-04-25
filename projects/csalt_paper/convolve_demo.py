import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from csalt.data import HDF_to_dataset
from parametric_disk_CSALT import *
from vis_sample import vis_sample
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

from matplotlib import font_manager

plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])

# Load the desired font
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


cfg = 'gen_fiducial_std_sspdemo1'
um, vm = np.array([400.]), np.array([400.])

do_create = False


if do_create:

    # Load the configuration file
    inp = importlib.import_module(cfg)

    # Ingest the template information into a dataset object
    tmp = HDF_to_dataset(inp.template_dir+inp.template[0])

    # Extract 'fixed' parameters of relevance
    restfreq, FOV, Npix, dist = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist
    fixed = restfreq, FOV, Npix, dist, inp.cfg_dict

    # Generate LSRK velocities for first/last integrations
    mid = np.int(tmp.nu_LSRK.shape[0] / 2)
    v_model_i = sc.c * (1 - tmp.nu_LSRK[mid,:] / restfreq)
    v_model_f = sc.c * (1 - tmp.nu_LSRK[mid,:] / restfreq)

    # Model cubes for each case
    mcube_i = parametric_disk(v_model_i, inp.pars, fixed)
    mcube_f = parametric_disk(v_model_f, inp.pars, fixed)

    # Convert spatial frequencies to lambda units
    uu = um * np.mean(tmp.nu_TOPO) / sc.c
    vv = vm * np.mean(tmp.nu_TOPO) / sc.c

    # Sample the FT of the cube onto the spatial frequency point
    mvis_i = vis_sample(imagefile=mcube_i, uu=uu, vv=vv, mod_interp=False)
    mvis_f = vis_sample(imagefile=mcube_f, uu=uu, vv=vv, mod_interp=False)
    mvis_i, mvis_f = np.squeeze(mvis_i), np.squeeze(mvis_f)

    # Simple output (ascii) of the data
    np.savetxt('convolve_demo_data.txt', list(zip(tmp.nu_TOPO, 
               tmp.nu_LSRK[mid,:], mvis_i.real, tmp.nu_LSRK[mid,:], 
               mvis_f.real)))


# Load data
nu_T, nu_i, vis_i, nu_f, vis_f = np.loadtxt('convolve_demo_data.txt').T


xlims = [-2.5, 2.5]
ylims = [-0.25, 0.15]
yylims = [-0.075, 0.075]
yrat = np.diff(ylims)[0] / np.diff(yylims)[0]

# Configure plot
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 3.45),
                        gridspec_kw={'height_ratios': [yrat, 1]}) 

foo = 20


# direct comparisons
ax = axs[0]

# "true" high-res spectrum
nu0 = 230.538e9 * (1 - 5000 / sc.c)
nu_true = nu_i
vis_true = vis_i

# SRF'ed
chix = np.arange(500)
xch = chix - np.mean(chix)
SRF_s = 0.5 * np.sinc(xch) + 0.25 * np.sinc(xch-foo) + 0.25 * np.sinc(xch+foo)
vis_s = convolve1d(vis_true, SRF_s / np.sum(SRF_s), mode='nearest')
vis_sim = vis_s[::foo]
nu_sim = nu_i[::foo]

nu_inf = nu_sim
SRF_i = np.array([0, 0.25, 0.5, 0.25, 0.])
vis_inf = convolve1d(vis_true[::foo], SRF_i / np.sum(SRF_i), mode='nearest')


ax.plot(1e-6 * (nu_inf - nu0), vis_inf, 'or', fillstyle='none', ms=3, zorder=3)
ax.plot(1e-6 * (nu_sim - nu0), vis_sim, '.k', ms=5)
ax.plot(1e-6 * (nu_true - nu0), vis_true, '-', color='silver', zorder=0)
ax.set_xlim(xlims)
ax.set_ylim(ylims)
#ax.set_xlabel('LSRK $\\nu - \\nu_{\\rm sys}$  (MHz)')
ax.set_ylabel('real  $\mathsf{V}^{\,}_{\\nu}$  (Jy)', 
              labelpad=5.5)

#ax.text(0.025, 0.20, 'fixed $(\mathsf{u}, \mathsf{v})$', transform=ax.transAxes,
#        ha='left', color='darkslategray')
ax.plot([0.025], [0.06], 'or', transform=ax.transAxes, ms=3, fillstyle='none')
ax.text(0.05, 0.04, 'inference', transform=ax.transAxes, ha='left', color='r',
        fontsize=8)
ax.plot([0.025], [0.14], '.k', transform=ax.transAxes, ms=5)
ax.text(0.05, 0.12, 'simulate', transform=ax.transAxes, ha='left', color='k',
        fontsize=8)


# secondary x-axis
def a2b(x):
    return -1e-6 * 230.538e9 * (x / sc.c)


def b2a(x):
    return -1e6 * (sc.c / x) / 230.538e9

secax = ax.secondary_xaxis('top', functions=(b2a, a2b))
secax.set_xlabel('LSRK $v - v_{\\rm sys}$  (m/s)', labelpad=7)



# the TOPO spectra from start and end of track
ax = axs[1]

resid = (vis_sim - vis_inf)
ax.plot(1e-6 * (nu_sim - nu0), resid, 'or', ms=3, fillstyle='none', zorder=3)
ax.plot(xlims, [0, 0], ':k')
ax.set_xlim(xlims)
ax.set_ylim(yylims)
ax.set_xlabel('LSRK $\\nu - \\nu_{\\rm sys}$  (MHz)')
ax.set_ylabel('$\Delta \mathsf{V}^{\,}_{\\nu}$  (Jy)', 
              labelpad=1)


fig.subplots_adjust(left=0.16, right=0.84, bottom=0.115, top=0.88, hspace=0.25)
fig.savefig('figs/convolve_demo.pdf')
fig.clf()
