"""
    This CASA script generates an observational template for synthetic data, 
    using the CASA.simobserve module.  It is called (usually inside the 
    csalt.synthesize.make_data() subroutine and an external CASA call) as

        execfile('generate_template.py <cfg_file> <EB>')  

    where <cfg_file> is the relevant part of the configuration input filename 
    (i.e., configs/gen_<cfg_file>.py) and <EB> is the appropriate index of the 
    execution block of interest.

    This script will output ...

"""

import os, sys
import numpy as np
import scipy.constants as sc
import h5py
execfile('csalt/CASA_scripts/ms_to_hdf5.py')


# Parse the arguments
cfg_file, EB = sys.argv[-2], np.int(sys.argv[-1])

# Load the configuration file
execfile('configs/gen_'+cfg_file+'.py')

# Move to template simulation space
os.chdir(template_dir)

# Get number of native channels needed to span the desired velocity range
nch = 2 * np.int(V_span[EB] / (sc.c * dnu_native[EB] / nu_rest)) + 1

# Get TOPO frequency corresponding to desired tuning velocity (center of SPW)
t0 = au.lstToUT(au.hoursToHMS(RAdeg / 15 + np.float((HA_0[EB])[:-1])), date[EB])
dt = t0[0][:-3].replace('-', '/').replace(' ','/')
nu_tune = au.restToTopo(nu_rest, 1e-3 * V_tune[EB], dt, RA, DEC)

# Generate a dummy cube
dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])  
res = ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)
ia.done()

# Simulate observations to generate the MS structure
os.chdir('sims')
simobserve(project=template[EB]+'.sim', skymodel='../dummy.image',
           antennalist=antcfg_dir+config[EB]+'.cfg', totaltime=ttotal[EB], 
           integration=tinteg[EB], thermalnoise='', hourangle=HA_0[EB],
           indirection='J2000 '+RA+' '+DEC, refdate=date[EB], 
           incell='0.01arcsec', mapsize='5arcsec', 
           incenter=str(nu_tune/1e9)+'GHz', 
           inwidth=str(dnu_native[EB]*1e-3)+'kHz', outframe='TOPO')
os.system('rm -rf *.last')
os.chdir('..')

# Move the template MS into template_dir
os.system('rm -rf '+template[EB]+'.ms*')
sim_MS = 'sims/'+template[EB]+'.sim/'+template[EB]+'.sim.'+config[EB]+'.ms'
os.system('mv '+sim_MS+' '+template[EB]+'.ms')

# Write MS file contents out to HDF5 format
ms_to_hdf5(template[EB], template[EB])

# Clean up
os.system('rm -rf dummy.image')
os.chdir('..')
os.system('rm -rf *.last')
