import os, sys, importlib
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
sys.path.append('../../')
sys.path.append('../../configs/')
from csalt.data import HDF_to_dataset
from parametric_disk_CSALT import *
from vis_sample import vis_sample
from scipy.ndimage import convolve1d
from scipy.interpolate import interp1d


# Load configuration inputs
cfg = 'simp2_flow'
inp = importlib.import_module('gen_'+cfg)

# Load the visibilities into a dataset object
data = HDF_to_dataset('../../'+inp.dataname+'_pure.DATA', grp='EB0/')


# Make plots of (u,v) locations for 3 distinct timestamps (start, niddle, end)
col = ['C1', 'C0', 'C3']
stmps = [0, np.int(data.nstamps/2), data.nstamps-1]
usel, vsel = np.empty(3), np.empty(3)
for i in range(len(stmps)):
    
    # extract (u, v) datapoints
    um, vm = data.um[data.tstamp == stmps[i]], data.vm[data.tstamp == stmps[i]]

    # locate spectral reference points
    if i == 0:
        uvreg = (um <= -1050) & (um >= -1250) & (vm >= 1000) & (vm <= 1100)
        loc = np.where(um == um[uvreg][0])[0][0]
    usel[i], vsel[i] = um[loc], vm[loc]

    # make the figure
    fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)

    # plot
    ax.plot(um, vm, 'ok', ms=5)
    ax.plot(-um, -vm, 'ok', ms=5)
    ax.plot(um[loc], vm[loc], 'o', ms=20, fillstyle='none', mew=5, color=col[i])

    # limits
    ax.set_xlim([2500, -2500])
    ax.set_ylim([-2500, 2500])
    ax.grid()
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    fig.savefig('figs/flow_uv_t'+f"{i:02d}"+'.png', dpi=300)
    fig.clf()


# Plot of (u,v) locations for all timestamps
fig, ax = plt.subplots(figsize=(7.5, 7.5), constrained_layout=True)

ax.plot(data.um, data.vm, 'ok', ms=5)
ax.plot(-data.um, -data.vm, 'ok', ms=5)

for i in range(3):
    ax.plot(usel[i], vsel[i], 'o', ms=20, fillstyle='none', mew=5, color=col[i])

ax.set_xlim([2500, -2500])
ax.set_ylim([-2500, 2500])
ax.grid()
ax.set_xticklabels([])
ax.set_yticklabels([])

fig.savefig('figs/flow_uv_allt.png', dpi=300)
fig.clf()



# Output spectra for these (usel, vsel) positions
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)
for i in range(len(usel)):

    # Get spectrum
    spec = data.vis[0,:,np.logical_and(data.um == usel[i], data.vm == vsel[i])]
    spec = np.squeeze(spec).real
    velax = sc.c * (1 - data.nu_LSRK[stmps[i],:] / inp.nu_rest)

    # Plot it
    ax = axs[i]
    ax.plot(velax, spec, 'o', ms=5, color=col[i])

    ax.set_xlim([-2000, 12000])
    ax.set_ylim([-0.02, 0.07])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.savefig('figs/spec_sim_final.png', dpi=300)
fig.clf()



# Initial spectra for simulator mode
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)
for i in range(len(usel)):
    # Extract 'fixed' parameters of relevance
    restfreq, FOV, Npix, dist = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist
    fixed = restfreq, FOV, Npix, dist, inp.cfg_dict

    # Generate up-sampled LSRK velocities 
    nu = np.interp(np.arange((data.nchan-1) * inp.nover + 1),
                   np.arange(0, data.nchan * inp.nover, inp.nover),
                   data.nu_LSRK[stmps[i],:])
    v_model = sc.c * (1 - nu / restfreq)

    # Model cubes for each case
    mcube = parametric_disk(v_model, inp.pars, fixed)

    # Convert spatial frequencies to lambda units
    uu = np.array([usel[i]]) * np.mean(data.nu_TOPO) / sc.c
    vv = np.array([vsel[i]]) * np.mean(data.nu_TOPO) / sc.c

    # Sample the FT of the cube onto the spatial frequency point
    mvis = vis_sample(imagefile=mcube, uu=uu, vv=vv, mod_interp=False)
    mvis = np.squeeze(mvis).real

    # Plot it
    ax = axs[i]
    ax.plot(v_model, mvis, 'o', ms=5, color=col[i])

    ax.set_xlim([-2000, 12000])
    ax.set_ylim([-0.02, 0.07])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.savefig('figs/spec_sim_init.png', dpi=300)
fig.clf()



# Convolved spectra for simulator mode
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)
for i in range(len(usel)):
    # Extract 'fixed' parameters of relevance
    restfreq, FOV, Npix, dist = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist
    fixed = restfreq, FOV, Npix, dist, inp.cfg_dict

    # Generate up-sampled LSRK velocities 
    nu = np.interp(np.arange((data.nchan-1) * inp.nover + 1),
                   np.arange(0, data.nchan * inp.nover, inp.nover),
                   data.nu_LSRK[stmps[i],:])
    v_model = sc.c * (1 - nu / restfreq)

    # Model cubes for each case
    mcube = parametric_disk(v_model, inp.pars, fixed)

    # Convert spatial frequencies to lambda units
    uu = np.array([usel[i]]) * np.mean(data.nu_TOPO) / sc.c
    vv = np.array([vsel[i]]) * np.mean(data.nu_TOPO) / sc.c

    # Sample the FT of the cube onto the spatial frequency point
    mvis = vis_sample(imagefile=mcube, uu=uu, vv=vv, mod_interp=False)
    mvis = np.squeeze(mvis).real

    # Convolve with SRF
    chix = np.arange(25 * inp.nover) / inp.nover
    xch = chix - np.mean(chix)
    SRF = 0.5*np.sinc(xch) + 0.25*np.sinc(xch-1) + 0.25*np.sinc(xch+1)
    mvis = convolve1d(mvis, SRF/np.sum(SRF), mode='nearest')

    # Plot it
    ax = axs[i]
    ax.plot(v_model, mvis, 'o', ms=5, color=col[i])

    ax.set_xlim([-2000, 12000])
    ax.set_ylim([-0.02, 0.07])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.savefig('figs/spec_sim_srf.png', dpi=300)
fig.clf()



# SRF simulator
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)

ax = axs[1]
chix = np.arange(25 * 15) / 15
xch = chix - np.mean(chix)
SRF = 0.5*np.sinc(xch) + 0.25*np.sinc(xch-1) + 0.25*np.sinc(xch+1)

ax.plot(5000 + xch * 158, SRF, 'k', lw=2)
ax.set_xlim([-2000, 12000])
ax.axis('off')

axs[0].axis('off')
axs[2].axis('off')

fig.savefig('figs/SRF_sim.png', dpi=300)




# Initial spectra for inference mode
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)
for i in range(len(usel)):
    # Extract 'fixed' parameters of relevance
    restfreq, FOV, Npix, dist = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist
    fixed = restfreq, FOV, Npix, dist, inp.cfg_dict

    # Generate up-sampled LSRK velocities 
    nu = data.nu_LSRK[stmps[1],:]
    v_model = sc.c * (1 - nu / restfreq)

    # Model cubes for each case
    mcube = parametric_disk(v_model, inp.pars, fixed)

    # Convert spatial frequencies to lambda units
    uu = np.array([usel[i]]) * np.mean(data.nu_TOPO) / sc.c
    vv = np.array([vsel[i]]) * np.mean(data.nu_TOPO) / sc.c

    # Sample the FT of the cube onto the spatial frequency point
    mvis = vis_sample(imagefile=mcube, uu=uu, vv=vv, mod_interp=False)
    mvis = np.squeeze(mvis).real

    # Plot it
    ax = axs[i]
    ax.plot(v_model, mvis, 'o', ms=5, color=col[i])

    ax.set_xlim([-2000, 12000])
    ax.set_ylim([-0.02, 0.07])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.savefig('figs/spec_inf_init.png', dpi=300)
fig.clf()



# Interpolated spectra for inference mode
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)
for i in range(len(usel)):
    # Extract 'fixed' parameters of relevance
    restfreq, FOV, Npix, dist = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist
    fixed = restfreq, FOV, Npix, dist, inp.cfg_dict

    # Generate up-sampled LSRK velocities 
    nu = data.nu_LSRK[stmps[1],:]
    v_model = sc.c * (1 - nu / restfreq)

    # Model cubes for each case
    mcube = parametric_disk(v_model, inp.pars, fixed)

    # Convert spatial frequencies to lambda units
    uu = np.array([usel[i]]) * np.mean(data.nu_TOPO) / sc.c
    vv = np.array([vsel[i]]) * np.mean(data.nu_TOPO) / sc.c

    # Sample the FT of the cube onto the spatial frequency point
    mvis = vis_sample(imagefile=mcube, uu=uu, vv=vv, mod_interp=False)
    mvis = np.squeeze(mvis).real

    # Interpolate
    if i != 1:
        vint = interp1d(v_model, mvis, fill_value='extrapolate', kind='cubic')
        mvis = vint(sc.c * (1 - data.nu_LSRK[stmps[i],:] / restfreq))

    # Plot it
    ax = axs[i]
    ax.plot(v_model, mvis, 'o', ms=5, color=col[i])

    ax.set_xlim([-2000, 12000])
    ax.set_ylim([-0.02, 0.07])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.savefig('figs/spec_inf_interp.png', dpi=300)
fig.clf()


# SRF-convolved spectra for inference mode
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)
for i in range(len(usel)):
    # Extract 'fixed' parameters of relevance
    restfreq, FOV, Npix, dist = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist
    fixed = restfreq, FOV, Npix, dist, inp.cfg_dict

    # Generate up-sampled LSRK velocities 
    nu = data.nu_LSRK[stmps[1],:]
    v_model = sc.c * (1 - nu / restfreq)

    # Model cubes for each case
    mcube = parametric_disk(v_model, inp.pars, fixed)

    # Convert spatial frequencies to lambda units
    uu = np.array([usel[i]]) * np.mean(data.nu_TOPO) / sc.c
    vv = np.array([vsel[i]]) * np.mean(data.nu_TOPO) / sc.c

    # Sample the FT of the cube onto the spatial frequency point
    mvis = vis_sample(imagefile=mcube, uu=uu, vv=vv, mod_interp=False)
    mvis = np.squeeze(mvis).real

    # Interpolate
    if i != 1:
        vint = interp1d(v_model, mvis, fill_value='extrapolate', kind='cubic')
        mvis = vint(sc.c * (1 - data.nu_LSRK[stmps[i],:] / restfreq))

    # Convolve with SRF
    SRF = np.array([0, 0.25, 0.5, 0.25, 0])
    mvis = convolve1d(mvis, SRF/np.sum(SRF), mode='nearest')

    # Plot it
    ax = axs[i]
    ax.plot(v_model, mvis, 'o', ms=5, color=col[i])

    ax.set_xlim([-2000, 12000])
    ax.set_ylim([-0.02, 0.07])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

fig.savefig('figs/spec_inf_srf.png', dpi=300)
fig.clf()



# SRF simulator
fig, axs = plt.subplots(figsize=(5.5, 9.5), ncols=1, nrows=3,
                        constrained_layout=True)

ax = axs[1]
chix = np.arange(25)
xch = chix - np.mean(chix)
SRF = np.zeros_like(xch)
SRF[xch == 0] = 0.5
SRF[xch == -1] = 0.25
SRF[xch == 1] = 0.25
ax.plot(5000 + xch * 158, SRF, 'k', lw=2)
ax.set_xlim([-2000, 12000])
ax.axis('off')

axs[0].axis('off')
axs[2].axis('off')

fig.savefig('figs/SRF_inf.png', dpi=300)








    



