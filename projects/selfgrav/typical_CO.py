import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc

# style setups
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import cmasher as cmr
from matplotlib import font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'

# Plot configuration
fig, ax = plt.subplots(figsize=(3.5, 2.32))
left, right, bottom, top = 0.15, 0.85, 0.16, 0.99

# Load data from Long et al. 2021
Jup, Fco, Rco, eRco, d, Mstar = np.loadtxt('../csalt_paper/data/feng_co.txt',
                                           usecols=(1,2,3,4,5,6), skiprows=1).T

# Proper adjustments for fluxes
F_co = 1.*Fco
F_co[Jup == 3] *= (230.538 / 345.796)**2

# select in a mstar range
mcond = np.logical_and(Mstar <= 2.0, Mstar >= 0.5)
L_CO = F_co[mcond] * (d[mcond] / 150)**2
eL_CO = 0.2 * L_CO
R_CO = Rco[mcond]
eR_CO = np.sqrt(eRco[mcond]**2 + (0.1 * R_CO)**2)

# plot Rco versus Lco
ax.errorbar(L_CO, R_CO, xerr=eL_CO, yerr=eR_CO, fmt='o', mec='C0', mfc='C0',
            elinewidth=1.5, ms=3.0, ecolor='C0', zorder=3)

# the sharp and taper models!
ax.plot(np.array([6.7]), np.array([275]), '*C1', ms=15, zorder=10)
ax.plot(np.array([11.]), np.array([395]), '*C3', ms=15, zorder=10)

ax.set_xlim([0.2, 50])
ax.set_xscale('log')
ax.set_xticks([1, 10])
ax.set_xticklabels(['1', '10'])
ax.set_xlabel('$F_{\\rm CO} \\times (d\,/\, 150\,{\\rm pc})^2$  (Jy km/s)')
ax.set_ylim([30, 2000])
ax.set_yscale('log')
ax.set_yticks([100, 1000])
ax.set_yticklabels(['100', '1000'])
ax.set_ylabel('$r_{\\rm CO}$  (au)')
ax.set_aspect('equal')

fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
fig.savefig('figs/typical_CO.pdf')
fig.clf()
