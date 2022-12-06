import os, sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(['default', {
    'font.size': 8.5,
    'xtick.top': True,
    'xtick.direction': 'in',
    'ytick.right': True,
    'ytick.direction': 'in',
    'mathtext.fontset': 'cm'
    }])


# Load data from Long et al. 2021
Jup, Fco, Rco, eRco, d, Mstar = np.loadtxt('../csalt_paper/data/feng_co.txt', 
                                           usecols=(1,2,3,4,5,6), skiprows=1).T


# MY MODELS
Fme = [5.8]
Rme = [250.]
Mme = [1.0]

# Proper adjustments for fluxes
F_co = Fco
F_co[Jup == 3] *= (230.538 / 345.796)**2


# Set up plots
fig, axs = plt.subplots(nrows=2, figsize=(3.5, 3.8))

# Mstar versus Fco
ax = axs[0]
ax.plot(Mstar, F_co * (d / 150)**2, marker='o', color='C0', mfc='C0', 
        linestyle='None', ms=3.0, fillstyle='full')
ax.plot(Mme, Fme, marker='*', ms=9.5, color='C1', mfc='C1', linestyle='None',
        fillstyle='full')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.08, 3.])
ax.set_ylim([0.05, 200])
ax.set_xticks([0.1, 1])
ax.set_xticklabels([])
ax.set_yticks([0.1, 1, 10, 100])
ax.set_yticklabels(['0.1', '1', '10', '100'])
ax.set_ylabel('$F_{\\rm CO} \\times (d / 150)^2 \,\,\, [{\\rm Jy \, km/s}]$', 
              fontsize=10.5, labelpad=5)


# Mstar versus Rco
ax = axs[1]
ax.errorbar(Mstar, Rco, yerr=eRco, fmt='o', mec='C0', mfc='C0', elinewidth=1.5,
            ms=3.0, ecolor='C0', zorder=3)
ax.plot(Mme, Rme, marker='*', ms=9.5, color='C1', mfc='C1', linestyle='None',
        fillstyle='full')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim([0.08, 3.])
ax.set_ylim([8, 2000])
ax.set_xticks([0.1, 1])
ax.set_xticklabels(['0.1', '1'])
ax.set_yticks([10, 100, 1000])
ax.set_yticklabels(['10', '100', '1000'])
ax.set_xlabel('$M_\\ast \,\,\, [{\\rm M}_\\odot]$', fontsize=10.5, labelpad=1)
ax.set_ylabel('$R_{\\rm CO} \,\,\, [{\\rm au}]$', fontsize=10.5, labelpad=0)


fig.subplots_adjust(hspace=0.05, left=0.15, right=0.85, bottom=0.09, top=0.99)
fig.savefig('figs/simple_model_justify.pdf') 

