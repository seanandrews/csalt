import os, sys, importlib
import numpy as np
import scipy.constants as sc
import emcee
import matplotlib.pyplot as plt
import pygtc


# Load the "snap" posterior samples
snap_file = '../../storage/posteriors/fidelity/fiducial_snap_pure.h5'
reader = emcee.backends.HDFBackend(snap_file)
snap_samples = reader.get_chain(discard=2000, flat=True)

# Load the "std" posterior samples
std_file = '../../storage/posteriors/fidelity/fiducial_std_pure.h5'
reader = emcee.backends.HDFBackend(std_file)
std_samples = reader.get_chain(discard=2000, flat=True)

# Re-organize parameter indexing (for aesthetics)
"""
    in =  incl, PA, mstar, r_l, z1, psi, T0, q, Tmaxb, sigma0, logtau0, p, 
          vsys, dx, dy

    out = z1, psi, r_l, T0, q, Tmaxb, logtau0, p, sigma0, mstar, incl, PA, 
          vsys, dx, dy
"""
ix = [4, 5, 3, 6, 7, 8, 10, 11, 9, 2, 0, 1, 12, 13, 14]
snap_ = np.take(snap_samples, ix, axis=-1)
std_ = np.take(std_samples, ix, axis=-1)

# Clip off nuisance parameters
snap = snap_[:,:-3]
std = std_[:,:-3]

# Units / scaling (for aesthetics)
snap[:,0] *= 150
std[:,0] *= 150
snap[:,4] *= -1
std[:,4] *= -1
snap[:,7] *= -1
std[:,7] *= -1

# Labeling
names = ['$z_1$', '$\psi$', '$r_\ell$', '$T_0$', '$q$', 
         '$T_{\\rm max}^{\, \\tt{b}}$', '$\log{\\tau_0}$', '$p$', 
         '$\sigma_0$', '$M_\\ast$', '$i$i', '$\\vartheta$']

# Ranges
paramRanges = ((35, 45), (1.0, 1.5), (245, 255), (120, 180), (0.4, 0.6),
               (15, 30), (1.0, 4.0), (0.0, 1.5), (250, 600), (0.9, 1.1),
               (38, 42), (128, 132))

# Make the plot
fig = pygtc.plotGTC(chains=[snap, std], 
                    paramNames=names,
                    paramRanges=paramRanges,
                    figureSize='APJ_page')

fig.subplots_adjust(left=0.09, right=0.91, bottom=0.10, top=0.99)

fig.savefig('figs/fidelity_cornerplot.pdf')
