import os, sys, importlib
import numpy as np
import scipy.constants as sc
import emcee
import matplotlib.pyplot as plt
import corner
from matplotlib import font_manager

plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])

# Load the desired font
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.labelsize'] = 7.0
plt.rcParams['xtick.major.size'] = 3.5
plt.rcParams['ytick.labelsize'] = 7.0
plt.rcParams['ytick.major.size'] = 3.5


# Inputs setup
pdir = '../../storage/posteriors/fidelity/'
in_files = ['../../storage/posteriors/noise/fiducial_std_noisy.h5',
            '../../storage/posteriors/fidelity/fiducial_std_pure.h5']
figtitle = 'noise'
burnins = [2000, 2000]
cbins = [50, 50]

post_lbls = ['noisy standard', 'pure standard']


# Output controls
# Labeling
lbls = ['$z_1$', '$\psi$', '$r_\ell$', '$T_0$', '$q$',
         '$T_{\\rm max}^{\, \\tt{b}}$', '$\log{\\tau_0}$', '$p$',
         '$\sigma_0$', '$M_\\ast$', '$i$', '$\\vartheta$']

# Ranges
paramRanges = [(35, 45), (1.0, 1.5), (245, 255), (125, 175), (0.4, 0.6),
               (15, 30), (1.0, 4.0), (0.0, 1.5), (250, 600), (0.9, 1.1),
               (37, 43), (127.5, 132.5)]

# Truths
truths = [0.2669 * 150., 1.25, 250., 150., 0.5, 20., np.log10(2000.), 1.0,
          297.3, 1.0, 40., 130.]
tcol = 'grey'

# Corner plot contour levels
levs = 1 - np.exp(-0.5 * np.array([1, 2, 3])**2)

# Colors
cols = ['C0', 'C1']


# Make the figure object
fig, axs = plt.subplots(12, 12, figsize=(7.5, 6.8))


# Model draws parameters
rau = np.logspace(-1, np.log10(truths[2]), 512)
Ndraws = 100
alp_draw = 0.02


# Loop through posterior sets
Nsamps = np.empty(len(in_files), dtype='int')
draws = np.empty((len(in_files), Ndraws, 12))
for ip in range(len(in_files)):

    # Load the posterior samples; discard burn-in; flatten chains
    reader = emcee.backends.HDFBackend(in_files[ip])
    _samps = reader.get_chain(discard=burnins[ip], flat=True)

    # Re-organize parameter indexing (for aesthetics)
    """
        in =  incl, PA, mstar, r_l, z1, psi, T0, q, Tmaxb, sigma0, logtau0, p, 
              vsys, dx, dy

        out = z1, psi, r_l, T0, q, Tmaxb, logtau0, p, sigma0, mstar, incl, PA, 
              vsys, dx, dy
    """
    ix = [4, 5, 3, 6, 7, 8, 10, 11, 9, 2, 0, 1, 12, 13, 14]
    samps_ = np.take(_samps, ix, axis=-1)

    # Clip off nuisance parameters
    samps_ = samps_[:,:-3]

    # Units / scaling (for aesthetics)
    samps_[:,0] *= 150
    samps_[:,4] *= -1
    samps_[:,7] *= -1

    # Record the number of samples
    Nsamps[ip] = samps_.shape[0]

    # Make the corner plot
    contour_kwargs = {'colors': cols[ip], 'linewidths': 1.0}
    corner.corner(samps_, levels=levs, range=paramRanges, bins=cbins[ip], 
                  max_n_ticks=3, 
                  weights=np.ones(Nsamps[ip]) * Nsamps[0] / Nsamps[ip],
                  color=cols[ip],
                  fill_contours=False, contour_kwargs=contour_kwargs,
                  fig=fig, plot_datapoints=False, plot_density=False)

    if ip == 0:
        # Axis labels (brute force)
        for i in np.arange(1, 12):
            axs[i][0].text(-0.7, 0.5, lbls[i], transform=axs[i][0].transAxes, 
                           ha='center', va='center', rotation=90, fontsize=8)
        for i in range(12):
            axs[11][i].text(0.5, -0.75, lbls[i], transform=axs[11][i].transAxes,
                            ha='center', va='center', fontsize=8)

        for iy in range(12):
            for ix in range(iy):
                if not (ix == 0):
                    axs[iy][ix].tick_params(axis='y', left=False)
        


    # Random posterior draws in sub-panels
    ix_draws = np.random.randint(0, high=Nsamps[ip], size=Ndraws)
    draws[ip,:,:] = samps_[ix_draws,:]


# Overplot truths in a more controllable format
corner.overplot_lines(fig, truths, color=tcol, linestyle='--', lw=1.0, zorder=0)



# Fake axis for annotations
axa = fig.add_axes([0.22, 0.85, 0.18, 0.14])
for ip in range(len(in_files)):
    axa.text(0.15, 0.88-0.20*ip, post_lbls[ip], transform=axa.transAxes,
             ha='left', va='center', fontsize=12, color=cols[ip])

axa.plot([0.15, 0.30], [0.88-0.20*(ip + 1), 0.88-0.20*(ip + 1)], '--', 
         color=tcol, lw=1., transform=axa.transAxes)
axa.text(0.35, 0.88-0.20*(ip + 1), '$inputs$', transform=axa.transAxes,
         ha='left', va='center', fontsize=11, color=tcol)
axa.axis('off')




# Assign the subplots
axz = fig.add_axes([0.53, 0.85, 0.18, 0.14])
axt = fig.add_axes([0.96-0.18, 0.85, 0.18, 0.14])
axu = fig.add_axes([0.53, 0.65, 0.18, 0.14])
axv = fig.add_axes([0.96-0.18, 0.65, 0.18, 0.14])


for ip in range(draws.shape[0]):

    for ir in range(Ndraws): 

        # z_l
        z_draw = draws[ip,ir,0] * (rau / 150.)**draws[ip,ir,1]
        axz.plot(rau, z_draw, '-'+cols[ip], alpha=alp_draw)

        # T
        T_draw = draws[ip,ir,3] * (rau / 10.)**-draws[ip,ir,4]
        axt.plot(rau, T_draw, '-'+cols[ip], alpha=alp_draw)

        # tau
        tau_draw = np.log10(10**draws[ip,ir,6] * (rau / 10)**-draws[ip,ir,7])
        axu.plot(rau, tau_draw, '-'+cols[ip], alpha=alp_draw)

        # sigma
        sig_draw = draws[ip,ir,8] * (rau / 10)**(-0.5 * draws[ip,ir,4])
        axv.plot(rau, sig_draw, '-'+cols[ip], alpha=alp_draw)
    

### Sub-panel truths, labeling, etc.
# z_l (r)
axz.plot(rau, truths[0] * (rau / 150.)**truths[1], '--', color=tcol, lw=1)
axz.set_xlim([0, 260])
axz.set_ylim([0, 80])
axz.set_xlabel('$r$  (au)', labelpad=2)
axz.set_ylabel('$z_\ell$  (au)')

# T (r)
axt.plot(rau, truths[3] * (rau / 10)**-truths[4], '--', color=tcol, lw=1)

# set up y-axis scaling
def forward(x):
    return x**(1/2)

def inverse(x):
    return x**2

axt.set_xlim([0, 260])
axt.set_xlabel('$r$  (au)', labelpad=2)
axt.set_yscale('function', functions=(forward, inverse))
axt.set_ylim([5, 200])
axt.set_ylabel('$T$  (K)')

# tau (r)
axu.plot(rau, np.log10(10**truths[6] * (rau / 10)**-truths[7]), '--',
         color=tcol, lw=1)
axu.set_xlim([0, 260])
axu.set_ylim([0.5, 4])
axu.set_xlabel('$r$  (au)', labelpad=2)
axu.set_ylabel('$\log{\\tau}$')

# sigma (r)
axv.plot(rau, truths[8] * (rau / 10)**(-0.5 * truths[4]), '--', 
         color=tcol, lw=1)
axv.set_xlim([0, 260])
axv.set_ylim([0, 800])
axv.set_xlabel('$r$  (au)', labelpad=2)
axv.set_ylabel('$\sigma_v$  (m/s)')



fig.subplots_adjust(bottom=0.075, top=0.99, left=0.065, right=0.96,
                    hspace=0.0, wspace=0.0)

fig.savefig('figs/'+figtitle+'_cornerplot.pdf')
fig.clf()
