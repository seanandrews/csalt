import os, sys, time, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
import emcee
import corner
from model_vexact import model_vphi
import matplotlib.pyplot as plt

# style setups
from matplotlib import cm, font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'



### USER INPUTS
# naming / labeling
mdl = 'taper2hi'
msetup = '_raw'
post_file = 'posteriors/'+mdl+msetup+'_offsetSig.posteriors.h5'
burnin = 500
lbls = ['$M_\\ast$ ($M_\\odot$)', '$M_{\\rm disk}$ ($M_\\odot$)', '$\log{f}$']
truths = [1.0, 0.1, None]

ndraws = 100


# fit range
fit_r = [0, 350]



### SETUPS
# load the "data" object
dat = np.load('data/surf_vphi_'+mdl+msetup+'.npz')

# extract the inferred velocity profile and the corresponding v_kep
re, ve, e_ve = dat['re'], dat['ve'], dat['eve']
ve_kep = dat['ve_kep']

# assign the subset to fit (and not)
ok = np.logical_and(re >= fit_r[0], re <= fit_r[1])
f_r, f_v, f_e = re[ok], ve[ok], e_ve[ok]
n_r, n_v, n_e = re[~ok], ve[~ok], e_ve[~ok]
print(f_e)

# assign the *true* vphi(r) and vkep(r) 
r_, vkep, vtot = dat['r_'], dat['v_kep'], dat['v_tot']


# load the posteriors
reader = emcee.backends.HDFBackend(post_file)
all_samples = reader.get_chain(discard=0, flat=False)
samples = reader.get_chain(discard=burnin, flat=False)
samples_ = reader.get_chain(discard=burnin, flat=True)
logpost_samples = reader.get_log_prob(discard=burnin, flat=False)
logprior_samples = reader.get_blobs(discard=burnin, flat=False)
nstep, nwalk, ndim = samples.shape
print(samples.shape)
print(samples_.shape)





### TRACES
fig, axs = plt.subplots(nrows=ndim+1, figsize=(3.5, 1.8 * (ndim + 1)),
                        constrained_layout=True)

# log-posterior
ax = axs[0]
for iw in range(nwalk):
    ax.plot(np.arange(nstep), logpost_samples[:,iw], color='k', alpha=0.03)
ax.set_xlim([0, nstep])
ax.set_ylabel('log prob')
ax.set_xticklabels([])

# cycle through parameters
for idim in range(ndim):
    ax = axs[1+idim]
    for iw in range(nwalk):
        ax.plot(np.arange(nstep), samples[:,iw,idim], color='k', alpha=0.03)
    ax.set_xlim([0, nstep])
    if idim != ndim-1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('MCMC steps')
    ax.set_ylabel(lbls[idim])


fig.savefig('figs/'+mdl+msetup+'.posterior_traces.png')
fig.clf()



### POSTERIOR DRAWS
fig, axs = plt.subplots(nrows=2, figsize=(3.5, 4.0))

# vphi(r)
ax = axs[0]

# truths!
#ax.plot(r_, 1e-3 * vkep, ':', color='C0')
ax.plot(r_, 1e-3 * vtot, '-', color='k', lw=1.5)

# fitted data
ax.errorbar(f_r, 1e-3 * f_v, 1e-3 * f_e, fmt='.', color='gray', ms=5)

# posterior draws
rix = np.random.randint(0, samples_.shape[0], ndraws)
pdraws = samples_[rix,:]
for i in range(len(rix)):
    v_draw = model_vphi(r_, pdraws[i,:2])
    ax.plot(r_, 1e-3 * v_draw, 'C1', alpha=0.03)


# limits, labels, annotations
ax.set_xlim([0, 300])
ax.set_ylim([1, 20])
ax.set_yscale('log')
ax.set_yticks([1, 10])
ax.set_yticklabels(['1', '10'])
ax.set_ylabel('$v_\\phi$  (km/s)', labelpad=7)


# non-keplerian delta vphi(r)
ax = axs[1]

# truths!
#ax.axhline(y=0, linestyle=':', color='k')
ax.plot(r_, vtot - vkep, '-', color='k', lw=1.5)

# fitted data
ax.errorbar(f_r, f_v - ve_kep[ok], f_e, fmt='.', color='gray', ms=5)

# draws!
for i in range(len(rix)):
    v_draw = model_vphi(r_, pdraws[i,:2])
    ax.plot(r_, v_draw - vkep, 'C1', alpha=0.03)


# limits, labels, annotations
ax.set_xlim([0, 300])
ax.set_ylim([-200, 200])
ax.set_xlabel('$r$  (au)')
ax.set_ylabel('$\delta v_{\phi}$  (m/s)', labelpad=-1)


fig.subplots_adjust(left=0.14, right=0.86, bottom=0.10, top=0.99, hspace=0.20)
fig.savefig('figs/'+mdl+msetup+'.posterior_draws.pdf')
fig.clf()



## CORNER PLOT
lbls = ['$M_{\\ast} \,\,\, (M_{\\odot})$', 
        '$M_{\\rm disk} \,\,\, (M_{\\odot})$']
truths = [1.0, 0.1]
fig = corner.corner(samples_[:,:2], labels=lbls, truths=truths)

fig.savefig('figs/'+mdl+msetup+'.corner.pdf')
fig.clf()


