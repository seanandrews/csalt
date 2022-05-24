import os, sys, importlib
import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import cmasher as cmr
sys.path.append('../../configs/')
sys.path.append('../../')
from parametric_disk_RADMC3D import parametric_disk as pardisk_radmc
from csalt.utils import *
from scipy.interpolate import interp1d, griddata


# constants
_msun = 1.9891e33
_AU = sc.au * 1e2
_mu = 2.37  
_mH = (sc.m_e + sc.m_p) * 1e3
_k  = sc.k * 1e7
_G  = sc.G * 1e3

r_lims, zr_lims, z_lims = [0, 300], [0.0, 0.6], [0, 100]



# Make a model 
cfg = 'radmc_std'
inp = importlib.import_module('gen_'+cfg)
fixed = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist, inp.cfg_dict
#foo = pardisk_radmc([0], inp.pars, fixed, struct_only=True)


# cylindrical grid
r_ = np.logspace(-1, np.log10(300), 256)
z_ = np.logspace(-2, np.log10(300), 512)
rr, zz = np.meshgrid(r_, z_)

# scale height function
def Hp(r):
    cs = np.sqrt(_k * inp.Tmid0 * (r / 10)**inp.qmid / (_mu * _mH))
    om = np.sqrt(_G * inp.mstar * _msun / (r * _AU)**3)
    return cs / om

# CO surface height
zco = inp.zrmax * Hp(r_) / _AU
print('    ')
print(inp.zrmax * Hp(150.) / (150. * _AU))
print('    ')

# Spherical grid CO surface curve
sphx = 1. * r_
sphy = np.arctan2(zco, sphx)


# T(r,z) in cylindrical space
def T_gas(r, z):
    r, z = np.atleast_1d(r), np.atleast_1d(z)
    Tmid, Tatm = inp.Tmid0 * (r / 10)**inp.qmid, inp.Tatm0 * (r / 10)**inp.qatm
    print(inp.Tmid0, inp.Tatm0)
    H = Hp(r) / _AU
    fz = 0.5 * np.tanh(((z / r) - inp.a_z * (H / r)) / (inp.w_z * (H / r))) + 0.5
    Tout = Tmid + fz * (Tatm - Tmid)
    return np.clip(Tout, a_min=0, a_max=1000)

Tcyl = T_gas(rr, zz)



# Plot T(r, z) in cylindrical space
plt.style.use('default')
plt.rcParams.update({'font.size': 12})
fig, ax = plt.subplots(constrained_layout=True)
im = ax.contourf(r_, z_, Tcyl, levels=np.linspace(0, 100, 21), cmap='plasma')
ax.contour(r_, z_, Tcyl, levels=[20])
ax.plot(r_, zco, '-k')
ax.set_xlim(r_lims)
ax.set_ylim(z_lims)
ax.set_xlabel('$r$  (AU)')
ax.set_ylabel('$z$  (AU)')
fig.colorbar(im, ax=ax, label='$T_{\\rm gas}$  (K)')
fig.savefig('figs/Tcyl.png')
fig.clf()

fig, ax = plt.subplots(constrained_layout=True)
ir0, ir1 = np.abs(r_ - 50).argmin(), np.abs(r_ - 150).argmin()
ax.plot(z_, Tcyl[:,ir0], 'C0')
ax.plot(z_, Tcyl[:,ir1], 'C1')
ax.set_xlim([0, 60])
ax.set_ylim([0, 100])
ax.set_xlabel('$z$  (AU)')
ax.set_ylabel('$T_{\\rm gas}$  (K)')
fig.savefig('figs/Tcyl_cuts.png')
fig.clf()


sys.exit()


# Plot T(R,THETA) in spherical space
_ = radmc_plotter(inp.radmcname, 'gas_temperature', xlims=r_lims, ylims=zr_lims,
                  zlevs=np.linspace(0, 100, 21), lbl='$T_{\\rm gas}$  (K)',
                  overlay=['gas_temperature'], olevs=[[20]], 
                  ofx=sphx, ofy=sphy, show_grid=False)


sys.exit()


r0 = 10 * _AU

# Set up the temperature structure function
def T_gas(r, z):
    r, z = np.atleast_1d(r), np.atleast_1d(z)
    Tmid = inpz.Tmid0 * (r / r0)**inpz.qmid
    Tatm = inpz.Tmid0 * (r / r0)**inpz.qatm
    H = np.sqrt(_k * Tmid / (_mu * _mH)) / omega_Kep(r, np.zeros_like(r))
    fz = 0.5 * np.tanh(((z / r) - inpz.a_z * (H / r)) / \
                       (inpz.w_z * (H / r))) + 0.5
    Tout = Tmid + fz * (Tatm - Tmid)
    return np.clip(Tout, a_min=0, a_max=1000)

def omega_Kep(r, z):
    return np.sqrt(_G * (inpz.mstar * _msun) / np.hypot(r, z)**3)


zg = np.linspace(0, 50)
plt.plot(zg, T_gas(100 * _AU, zg * _AU))
plt.show()


sys.exit()




# Load temperature structure
#rsph, zsph, Tsph = radmc_loader(inp.radmcname, 'gas_temperature')
#rrsph, zzsph = np.meshgrid(rsph, zsph)

# Compute z_CO surface in cylindrical coordinates on arbitrary radial grid
r_ = np.logspace(-1, np.log10(300), 256)

def Hp(r):
    cs = np.sqrt(sc.k * inpi.Tmid0 * (r / 10)**inpi.qmid / \
                 (2.37 * (sc.m_p + sc.m_e)))
    om = np.sqrt(sc.G * inpi.mstar * 1.9891e30 / (r * sc.au)**3)
    return cs / om

zco = inpi.zrmax * Hp(r_) / sc.au
print('    ')
print(inpi.zrmax * Hp(150.) / (150. * sc.au))
print('    ')

# Convert spherical grid to (unstructured) cylindrical grid
#rcyl, zcyl = rrsph * np.cos(zzsph), rrsph * np.sin(zzsph)
ox = 1. * r_
oy = np.arctan2(zco, ox)

# 2-D interpolation of temperature profile in CO surface
#pts = np.column_stack((rcyl.flatten(), zcyl.flatten()))
#interps = np.column_stack((r_, zco))
#Tco = griddata(pts, Tsph.flatten(), interps, method='linear')


r0 = 10 * _AU

# Set up the temperature structure function
def T_gas(r, z):
    r, z = np.atleast_1d(r), np.atleast_1d(z)
    Tmid, Tatm = inp.Tmid0 * (r / r0)**inp.qmid, inp.Tatm0 * (r / r0)**inp.qatm
    H = np.sqrt(_k * Tmid / (_mu * _mH)) / omega_Kep(r, np.zeros_like(r))
    fz = 0.5 * np.tanh(((z / r) - inp.a_z * (H / r)) / (inp.w_z * (H / r))) + 0.5
    Tout = (Tmid**4 + fz * Tatm**4)**0.25
    return np.clip(Tout, a_min=0, a_max=1000)

# Set up the surface density function
def Sigma_gas(r):
    sig = inp.Sig0 * (r / r0)**inp.p1 * np.exp(-(r / (inp.r_l * _AU))**inp.p2)
    return np.clip(sig, a_min=1e-50, a_max=1e50)


def omega_Kep(r, z):
    return np.sqrt(_G * (inp.mstar * _msun) / np.hypot(r, z)**3)

def abund(r, z):
    H_p = np.sqrt(_k * inp.Tmid0 * (r / r0)**inp.qmid / (_mu * _mH)) / \
          omega_Kep(r, np.zeros_like(r))
    z_mask = np.logical_and(z <= inp.zrmax * H_p, T_gas(r, z) >= inp.Tfrz)
    r_mask = np.logical_and(r >= (inp.rmin * _AU), r <= (inp.rmax * _AU))
    return np.where(np.logical_and(z_mask, r_mask), inp.xmol, inp.xmol * inp.depl)

# load the gridpoints
_ = np.loadtxt(inp.radmcname+'/amr_grid.inp', skiprows=5, max_rows=1)
nr, nt = np.int(_[0]), np.int(_[1])
Rw = np.loadtxt(inp.radmcname+'/amr_grid.inp', skiprows=6, max_rows=nr+1)
Tw = np.loadtxt(inp.radmcname+'/amr_grid.inp', skiprows=nr+7, max_rows=nt+1)
xx = 0.5*(Rw[:-1] + Rw[1:]) 
yy = 0.5*(Tw[:-1] + Tw[1:])
rr, tt = np.meshgrid(xx, yy)
rc, zc = rr * np.sin(tt), rr * np.cos(tt)

xco = abund(rc, zc)

diffxco = np.diff(xco, axis=0)

surfco = np.zeros(len(xx)-1)
for ir in [120]:	#range(len(surfco)):
    print(ir, xx[ir], diffxco[:,ir].max())
    print(0.5 * np.pi - yy[np.where(diffxco[:,ir] == diffxco[:,ir].max())][0])
    #surfco[ir] = 0.5 * np.pi - yy[diffxco[:,ir] == diffxco[:,ir].max()].max()




_ = radmc_plotter(inp.radmcname, 'gas_temperature', xlims=r_lims, ylims=zr_lims,
                  zlevs=np.linspace(0, 100, 21), lbl='$T_{\\rm gas}$  (K)',
                  overlay=['numberdens_co', 'gas_temperature'], 
                  olevs=[[0.001], [20]], ofx=ox, ofy=oy,
                  show_grid=True)
