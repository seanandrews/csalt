import os, sys
from csalt.data import *

# I/O
datafile = 'storage/data/radmc_std/radmc_std_noisy.DATA'
post_dir = 'storage/posteriors/radmc/'
postfile = 'radmc_std_noisy.h5'

# model setups
mtype = 'RADMC'
mode  = 'iter'
vra_fit = [-1e3, 11e3]	# (LSRK velocities from -1 to +11 km/s)
vcensor = None

# inference setups
nwalk = 75
ninits = 300
nsteps = 5000
nthreads = 6

# fixed parameters
nu_rest = 230.538e9	# rest frequency of line (Hz)
FOV = 6.375		# field of view (arcsec)
Npix = 256		# pixels per side
dist = 150.		# distance (pc)
cfg_dict = {}		# empty dict for CSALT models
fixed = nu_rest, FOV, Npix, dist, cfg_dict




# Set up the prior functions
if not os.path.exists('priors_'+mtype+'.py'):
    print('There is no such file "priors_"+mtype+".py".\n')
    sys.exit()
else:
    os.system('rm priors.py')
    os.system('cp priors_'+mtype+'.py priors.py')

# If necessary, make the posteriors directory
if not os.path.exists(post_dir):
    os.system('mkdir '+post_dir)

# Run the inference
from csalt.fit import *
run_emcee(datafile, fixed, vra=vra_fit, vcensor=vcensor, nwalk=nwalk, 
          ninits=ninits, nsteps=nsteps, outfile=post_dir+postfile, mode=mode,
          nthreads=nthreads, append=True)
