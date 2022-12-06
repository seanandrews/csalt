import os, sys, time, importlib
import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
from structure_functions import *
from model_vexact import model_vphi
sys.path.append('../../configs/')
import matplotlib.pyplot as plt

mdl = 'taper2hi'

# import dictionary
inp = importlib.import_module('gen_sg_'+mdl)

# radial bins of interest
beam = np.sqrt(0.117 * 0.100)
rbins_vphi = np.arange(0.05, 1.9, 0.25 * beam)

# constants
mu = 2.37
mH = sc.m_e + sc.m_p

# spatial grid
r_, z_ = np.linspace(0.5, 300.5, 301), np.logspace(-1.5, 2, 501)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

# EPSILON_P
#epsilonP = model_vphi(rbins_vphi, [1.0, 0.1])
epsilonP_ = eps_P(rr, zz, inp, nselfgrav=True)
#diffp = (epsilonP_ - epsilonP) / epsilonP_
#print('min/max delta(epsilon_P) = %s, %s' % (diffp.min(), diffp.max()))

#ix = 10
#fig, ax = plt.subplots()
#print(r_[ix])
#ax.plot(z_, epsilonP_[:,ix], 'C0')
#ax.plot(z_, epsilonP[:,ix], '--C1')
#ax.set_xscale('log')
#plt.show()


# VELOCITY FIELD: ROTN + PRESSURE SUPPORT
#omtot = np.sqrt(model_vphi(rbins_vphi, [1.0, 0.1]))
omtot_ = np.sqrt(omega_kep(rr, zz, inp)**2 + epsilonP_ + eps_g(rr, zz, inp))
#diffo = (omtot_ - omtot) / omtot_
#print('min/max delta(omega) = %s, %s' % (diffo.min(), diffo.max()))

#ix = 100
#fig, ax = plt.subplots()
#print(r_[ix])
#ax.plot(z_, omtot_[:,ix], 'C0')
#ax.plot(z_, omtot[:,ix], '--C1')
#ax.set_xscale('log')
#plt.show()


# INTERPOLATION ONTO CO SURFACE
def zCO_func(r, pars, dpc=150):
    zco = dpc*pars[0] * (r/dpc)**pars[1] * np.exp(-(r/(dpc*pars[2]))**pars[3])
    return zco

#psurf = np.array([0.29723454, 1.28063122, 2.17227701, 5.60994904])
psurf = np.load('data/'+mdl+'.zCO_true.npz')['psurf']
zCOr = zCO_func(r_, psurf, dpc=150)


om_totr_ = np.empty_like(zCOr)
for ir in range(len(zCOr)):
    gavint_ = interp1d(z_, omtot_[:,ir], fill_value='extrapolate')
    om_totr_[ir] = gavint_(zCOr[ir])

vphir_ = r_ * sc.au * om_totr_

rint_ = interp1d(r_, vphir_, fill_value='extrapolate')
vphi_ = rint_(rbins_vphi * 150)

vphi = model_vphi(rbins_vphi * 150, [1.0, 0.1])


diffv = (vphi_ - vphi) / vphi_
print('min/max delta(v_r) = %s, %s' % (diffv.min(), diffv.max()))

plt.plot(rbins_vphi * 150, vphi_, 'C0')
plt.plot(rbins_vphi * 150, vphi, '--C1')
#plt.loglog(r_, omtot_[100,:], 'C0')
#plt.loglog(r_, omtot[100,:], '--C1')
plt.show()
