import os, sys
from csalt.data import *

# setups
datafile = 'storage/data/radmc2_slm/radmc2_slm_pure.DATA'
posteriors_file = 'storage/posteriors/radmc2_slm/radmc2_slm_pure.h5'
mtype = 'CSALT'
vra_fit = [0, 1e4]	# (LSRK velocities from 0 to 10 km/s)

nu_rest = 230.538e9	# rest frequency of line (Hz)
FOV = 6.375
Npix = 256
dist = 150.
cfg_dict = {}
nwalk = 60



# Set up priors functions; load fitting routines
if not os.path.exists('priors_'+mtype+'.py'):
    print('There is no such file "priors_"+mtype+".py".\n')
    sys.exit()
#else:
#    os.system('cp priors_'+mtype+'.py priors.py')
from csalt.fit import *


fixed = nu_rest, FOV, Npix, dist, cfg_dict
foo = run_emcee(datafile, fixed, vra=vra_fit, nwalk=nwalk, nsteps=5000,
                ninits=300, outfile=posteriors_file)


