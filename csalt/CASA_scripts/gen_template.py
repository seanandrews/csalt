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


"""
    Load information and prepare to generate simulation.
"""
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


"""
    Make the template (u, v) tracks (based on a "dummy" cube), as placeholders 
    for the real model (injected in the csalt.synthesize.make_data() module).
"""
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


"""
    Extract the relevant information from the template measurement set and 
    record it (in HDF5 format) for easier use in csalt infrastructure.
"""
# Acquire MS information (for easier use external to CASA)
tb.open(template[EB]+'.ms')
data = np.squeeze(tb.getcol("DATA"))
u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
weights = tb.getcol('WEIGHT')
times = tb.getcol("TIME")
tb.close()

# Index the timestamps
tstamps = np.unique(times)
tstamp_ID = np.empty_like(times)
for istamp in range(len(tstamps)):
    tstamp_ID[times == tstamps[istamp]] = istamp

# Acquire the TOPO frequency channels (Hz)
tb.open(template[EB]+'.ms/SPECTRAL_WINDOW')
nu_TOPO = np.squeeze(tb.getcol('CHAN_FREQ'))
tb.close()

# Compute the LSRK frequencies (Hz) for each timestamp
nu_LSRK = np.empty((len(tstamps), len(nu_TOPO)))
ms.open(template[EB]+'.ms')
for istamp in range(len(tstamps)):
    nu_LSRK[istamp,:] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                     obstime=str(tstamps[istamp])+'s')
ms.close()

# Record the results in HDF5 format
os.system('rm -rf '+template[EB]+'.h5')
outp = h5py.File(template[EB]+'.h5', "w")
outp.create_dataset("um", u.shape, dtype="float64")[:] = u
outp.create_dataset("vm", v.shape, dtype="float64")[:] = v
outp.create_dataset("vis_real", data.shape, dtype="float64")[:,:,:] = data.real
outp.create_dataset("vis_imag", data.shape, dtype="float64")[:,:,:] = data.imag
outp.create_dataset("weights", weights.shape, dtype="float64")[:,:] = weights
outp.create_dataset("nu_TOPO", nu_TOPO.shape, dtype="float64")[:] = nu_TOPO
outp.create_dataset("nu_LSRK", nu_LSRK.shape, dtype="float64")[:,:] = nu_LSRK
outp.create_dataset("tstamp_ID", tstamp_ID.shape, dtype="int")[:] = tstamp_ID
outp.close()

# Clean up
os.system('rm -rf dummy.image')
os.chdir('..')
os.system('rm -rf *.last')
