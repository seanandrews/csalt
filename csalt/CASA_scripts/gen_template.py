"""
CASA script to generate an observational template for synthetic data.  

Settings are controlled in 'tconfig_<template_name>.py'.

Usage: 
	CASA> execfile('mock_obs.py <template_name>')
"""

import os, sys
import numpy as np
import scipy.constants as sc


### Preparations
# Load configuration file
execfile('configs_synth/sconfig_'+sys.argv[-2]+'.py')

# Assign template index and prepare for simulation
ix = np.int(sys.argv[-1])
os.chdir(template_dir)



### Parse simulated observation inputs
# Number of native channels needed to span the desired velocity range
nch = 2 * np.int(V_span[ix] / (sc.c * dnu_native[ix] / nu_rest)) + 1

# TOPO frequency corresponding to desired tuning velocity (center of SPW)
t0 = au.lstToUT(au.hoursToHMS(RAdeg / 15 + np.float((HA_0[ix])[:-1])), date[ix])
dt = t0[0][:-3].replace('-', '/').replace(' ','/')
nu_tune = au.restToTopo(nu_rest, 1e-3 * V_tune[ix], dt, RA, DEC)



### Generate a dummy cube
dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])  
res = ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)
ia.done()



### Simulate observations to generate the MS structure
os.chdir('sims')
simobserve(project=template[ix]+'.sim', skymodel='../dummy.image',
           antennalist=simobs_dir+'alma.cycle7.'+config[ix]+'.cfg',
           totaltime=ttotal[ix], integration=tinteg[ix], thermalnoise='', 
           indirection='J2000 '+RA+' '+DEC, refdate=date[ix], 
           hourangle=HA_0[ix], incell='0.02arcsec', mapsize='10arcsec', 
           incenter=str(nu_tune/1e9)+'GHz', 
           inwidth=str(dnu_native[ix] * 1e-3)+'kHz', outframe='TOPO')
os.system('rm -rf *.last')
os.chdir('..')

# Move the template MS into template_dir
os.system('rm -rf '+template[ix]+'.ms*')
sim_MS = 'sims/'+template[ix]+'.sim/'+\
         template[ix]+'.sim.alma.cycle7.'+config[ix]+'.ms'
os.system('mv '+sim_MS+' '+template[ix]+'.ms')



### Acquire MS information (for easier use external to CASA)
# Basic MS contents
tb.open(template[ix]+'.ms')
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

# TOPO frequency channels (Hz)
tb.open(template[ix]+'.ms/SPECTRAL_WINDOW')
nu_TOPO = np.squeeze(tb.getcol('CHAN_FREQ'))
tb.close()

# LSRK frequencies (Hz) for each timestamp
nu_LSRK = np.empty((len(tstamps), len(nu_TOPO)))
ms.open(template[ix]+'.ms')
for istamp in range(len(tstamps)):
    nu_LSRK[istamp,:] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                     obstime=str(tstamps[istamp])+'s')
ms.close()



### Record the results
np.savez_compressed(template[ix]+'.npz', 
                    um=u, vm=v, data=data, weights=weights, 
                    nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK, tstamp_ID=tstamp_ID) 


# Clean up
os.system('rm -rf dummy.image')
os.chdir('..')
os.system('rm -rf *.last')
