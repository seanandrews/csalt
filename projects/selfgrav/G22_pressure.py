import os, sys
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr

# parameters
T0, q = 150., 0.5
Md, p, rout = 0.1, 1.0, 300
Mstar = 1.0
r0 = 10.

# additional constants
Msun = 1.9891e30
mu = 2.37
mH = sc.m_e + sc.m_p


# define spatial grid
r_, z_ = np.linspace(1.1, 301.1, 1201), np.linspace(0, 150, 601)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)
ext = (r_.min(), r_.max(), z_.min(), z_.max())

# define the 2-D temperature structure (vertically isothermal)
Tgas = T0 * (rr / (r0 * sc.au))**-q

# define the 2-D sound speed structure
cs = np.sqrt(sc.k * Tgas / (mu * mH))

# define the surface density profile
Sig0 = (2 - p) * Md * Msun / \
       (2 * np.pi * (r0 * sc.au)**p * (rout * sc.au)**(2 - p))
Sigma = Sig0 * (r_ / r0)**-p

# define the Keplerian angular frequency: Omega_kep = sqrt(G M / r**3)
om_kep = np.sqrt(sc.G * Mstar * Msun / rr**3)

# define the scale heights
Hp = cs / om_kep

# un-normalized 2-D density distribution
rho_ = np.exp(-(rr**2 / Hp**2) * (1. - 1./np.sqrt(1. + (zz**2/rr**2))))

# density normalization
rho0 = 0.5 * Sigma / np.trapz(rho_, z_ * sc.au, axis=0)

# normalized 2-D density distribution
rho = rho0 * rho_

# 2-D pressure distribution
Pgas = rho * cs**2

# pressure support term (here in velocity-space, not angular velocity-space)
# GL calls this v_p**2, but I prefer not to denote it as a squared value 
# because it is often negative (meaning then that v_p is imaginary!)
eps_P = (rr / rho) * np.gradient(Pgas, r_ * sc.au, axis=1)

fig, ax = plt.subplots()

im = ax.imshow(eps_P, origin='lower', cmap='bwr', extent=ext, aspect='auto',
               vmin=eps_P.min(), vmax=np.abs(eps_P.min()))

ax.plot(r_, Hp[0,:] / sc.au, '--k')
ax.plot(r_, 2 * Hp[0,:] / sc.au, '--k')
ax.plot(r_, 0.1 * r_, ':', color='gray')
ax.plot(r_, 0.2 * r_, ':', color='gray')
ax.plot(r_, 0.3 * r_, ':', color='gray')

ax.set_xlim([0, 300])
ax.set_ylim([0, 105])
ax.set_xlabel('$r$ (au)')
ax.set_ylabel('$z$ (au)')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax, label='$r/\\rho \,\,\,\,\, \partial P/\partial r$')

fig.savefig('me_numerical.png')



### Let's look at their analytical approximation of v_p**2, as written in 
### their Equation (A7):

# v_K = Omega_K * r 
v_K = om_kep * rr

# gam = 1.5 + p + 0.5*q
gam = 1.5 + p + 0.5 * q

# dlogH/dlogr = 1.5 - 0.5q
dlogHdlogr = 1.5 - 0.5 * q

# equation A7
vp2 = v_K**2 * (-gam * (Hp / rr)**2 + (2 / (1 + zz**2/rr**2)**1.5) * \
      (1 + (1.5 * zz**2 / rr**2) - (1 + zz**2 / rr**2)**1.5 - \
       dlogHdlogr * (1 + (zz**2 / rr**2) - (1 + (zz**2 / rr**2))**1.5)))

fig, ax = plt.subplots()

im = ax.imshow(vp2, origin='lower', cmap='bwr', extent=ext, aspect='auto',
               vmin=eps_P.min(), vmax=np.abs(eps_P.min()))

ax.plot(r_, Hp[0,:] / sc.au, '--k')
ax.plot(r_, 2 * Hp[0,:] / sc.au, '--k')
ax.plot(r_, 0.1 * r_, ':', color='gray')
ax.plot(r_, 0.2 * r_, ':', color='gray')
ax.plot(r_, 0.3 * r_, ':', color='gray')

ax.set_xlim([0, 300])
ax.set_ylim([0, 105])
ax.set_xlabel('$r$ (au)')
ax.set_ylabel('$z$ (au)')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax, label='$r/\\rho \,\,\,\,\, \partial P/\partial r$')

fig.savefig('GL22_analytic.png')


### equations (15) and (16) together!

# dvp (15)
dvp2 = -v_K**2 * (gam + (2 - p)) * (Hp / rr)**2

# dvz (16)
dvz2 = -q * v_K**2 * (1 - 1 / (np.sqrt(1 + (zz**2 / rr**2))))

fig, ax = plt.subplots()

im = ax.imshow(dvp2+dvz2, origin='lower', cmap='bwr', extent=ext, aspect='auto',
               vmin=eps_P.min(), vmax=np.abs(eps_P.min()))

ax.plot(r_, Hp[0,:] / sc.au, '--k')
ax.plot(r_, 2 * Hp[0,:] / sc.au, '--k')
ax.plot(r_, 0.1 * r_, ':', color='gray')
ax.plot(r_, 0.2 * r_, ':', color='gray')
ax.plot(r_, 0.3 * r_, ':', color='gray')

ax.set_xlim([0, 300])
ax.set_ylim([0, 105])
ax.set_xlabel('$r$ (au)')
ax.set_ylabel('$z$ (au)')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax, label='$r/\\rho \,\,\,\,\, \partial P/\partial r$')

fig.savefig('GL22_decomposed.png')



# my own version of A10
vrot2 = v_K**2 * (1 - gam * (Hp / rr)**2 - q * (1 - 1 / np.sqrt(1 + (zz/rr)**2)))

vp2_ = vrot2 - (v_K**2 / (1 + (zz/rr)**2)**1.5)

fig, ax = plt.subplots()

im = ax.imshow(vp2_, origin='lower', cmap='bwr', extent=ext, aspect='auto',
               vmin=eps_P.min(), vmax=np.abs(eps_P.min()))

ax.plot(r_, Hp[0,:] / sc.au, '--k')
ax.plot(r_, 2 * Hp[0,:] / sc.au, '--k')
ax.plot(r_, 0.1 * r_, ':', color='gray')
ax.plot(r_, 0.2 * r_, ':', color='gray')
ax.plot(r_, 0.3 * r_, ':', color='gray')

ax.set_xlim([0, 300])
ax.set_ylim([0, 105])
ax.set_xlabel('$r$ (au)')
ax.set_ylabel('$z$ (au)')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(im, cax=cax, label='$vrot^2$')

fig.savefig('me8_13.png')


