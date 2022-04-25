import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from csalt.data import HDF_to_dataset
from parametric_disk_CSALT import *
from vis_sample import vis_sample
import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])

# Load the desired font
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'



cfg = 'gen_fiducial_std_sspdemo1'
um, vm = np.array([200.]), np.array([200.])

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
    v_model_i = sc.c * (1 - tmp.nu_LSRK[0,:] / restfreq)
    v_model_f = sc.c * (1 - tmp.nu_LSRK[-1,:] / restfreq)

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
    np.savetxt('setting_demo_data.txt', list(zip(tmp.nu_TOPO, tmp.nu_LSRK[0,:],
               mvis_i.real, tmp.nu_LSRK[-1,:], mvis_f.real)))


# Load data
nu_T, nu_i, vis_i, nu_f, vis_f = np.loadtxt('setting_demo_data.txt').T


# Configure plot
#plt.rcParams.update({'font.size': 8})
fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(3.5, 4.5))

xlims = [-2.5, 2.5]


# the "true" LSRK visibility spectrum
ax = axs[0]

nu0 = 230.538e9 * (1 - 5000 / sc.c)
nu_ = np.concatenate((nu_i, nu_f))
vis_true = np.concatenate((vis_i, vis_f))[np.argsort(nu_)]
nu_true = np.sort(nu_)

ax.plot(1e-6 * (nu_i[::20] - nu0), vis_i[::20], '.C0', ms=5)
ax.plot(1e-6 * (nu_f[::20] - nu0), vis_f[::20], '.C1', ms=5)
ax.plot(1e-6 * (nu_true - nu0), vis_true, '-', color='silver', zorder=0)
ax.set_xlim(xlims)
ax.set_xlabel('LSRK $\\nu - \\nu_{\\rm sys}$  (MHz)')
ax.set_ylabel('real  $\mathsf{V}^{\, \prime}_{\\nu}$  (Jy)', labelpad=0)

ax.text(0.025, 0.12, 'fixed $(\mathsf{u}, \mathsf{v})$', transform=ax.transAxes,
        ha='left', color='darkslategray', fontsize=8)
ax.text(0.025, 0.04, 'HA = ', transform=ax.transAxes, ha='left', 
        color='darkslategray', fontsize=8)
ax.text(0.14, 0.04, '0 h', transform=ax.transAxes, ha='left', color='C0',
        fontsize=8)
ax.text(0.21, 0.04, ',', transform=ax.transAxes, ha='left', 
        color='darkslategray')
ax.text(0.23, 0.04, '1 h', transform=ax.transAxes, ha='left', color='C1',
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

nuT_0 = np.mean(nu_T)

ax.plot(1e-6 * (nu_T[::20] - nuT_0), vis_i[::20], '.C0', ms=5)
ax.plot(1e-6 * (nu_T[::20] - nuT_0), vis_f[::20], '.C1', ms=5)
ax.set_xlim(xlims)
ax.set_xlabel('TOPO $\\nu - \\nu_{\\rm sys}$  (MHz)')
ax.set_ylabel('real  $\mathsf{V}^{\, \prime}_{\\nu}$  (Jy)', labelpad=0)

ax.text(0.025, 0.12, 'fixed $(\mathsf{u}, \mathsf{v})$', transform=ax.transAxes,
        ha='left', color='darkslategray', fontsize=8)
ax.text(0.025, 0.04, 'HA = ', transform=ax.transAxes, ha='left',        
        color='darkslategray', fontsize=8)
ax.text(0.14, 0.04, '0 h', transform=ax.transAxes, ha='left', color='C0',
        fontsize=8)
ax.text(0.21, 0.04, ',', transform=ax.transAxes, ha='left', 
        color='darkslategray', fontsize=8)
ax.text(0.23, 0.04, '1 h', transform=ax.transAxes, ha='left', color='C1',
        fontsize=8)

fig.subplots_adjust(bottom=0.09, top=0.90, left=0.14, right=0.86, hspace=0.30)
fig.savefig('figs/setting_demo.pdf')
fig.clf()
