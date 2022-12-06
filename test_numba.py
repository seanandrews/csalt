import os, sys, time, importlib
sys.path.append('configs/')
import numpy as np
import scipy.constants as sc
from parametric_disk_CSALT import parametric_disk as pardisk_csalt
from csalt.data import *
from vis_sample import vis_sample as def_vs
sys.path.append('nvis_sample')
from nvis_sampler import vis_sample as num_vs

# random set of velocities
velax = np.arange(0, 10000, 100)

# load the config file
inp = importlib.import_module('gen_fiducial_std')

# prep inputs
fixed = inp.nu_rest, inp.FOV[0], inp.Npix[0], inp.dist, inp.cfg_dict


# load data
data_ = fitdata('storage/data/fiducial_std/fiducial_std_pure.DATA', 
                vra=[-3000, 13000], vcensor=None, nu_rest=inp.nu_rest, chbin=1)


# u, v points
uu = data_['0'].um * np.mean(data_['0'].nu_TOPO) / sc.c
vv = data_['0'].vm * np.mean(data_['0'].nu_TOPO) / sc.c


# compute a cube object
v_model = sc.c * (1 - data_['0'].nu_LSRK[0,:] / inp.nu_rest)
tc0 = time.time()
cubec = pardisk_csalt(velax, inp.pars, fixed)
tc1 = time.time()
print('\nElapsed time to generate cube = %s\n' % (tc1-tc0))


# FFT the cube with Ryan's vis_sample
blah = def_vs(imagefile=cubec, uu=uu, vv=vv, mod_interp=False)
tr0 = time.time()
mvis_ryanl = def_vs(imagefile=cubec, uu=uu, vv=vv, mod_interp=False)
tr1 = time.time()
print('Elapsed time for FFT with default vis_sample = %s\n' % (tr1-tr0))


# FFT the cube with a NUMBA version of vis_sample
tn0 = time.time()
mvis_numba = num_vs(imagefile=cubec, uu=uu, vv=vv, mod_interp=False)
tn1 = time.time()
print('Elapsed time for FFT with NUMBA vis_sample = %s\n' % (tn1-tn0))

