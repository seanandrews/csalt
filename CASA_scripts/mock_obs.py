"""
CASA script to generate an observational template for synthetic data.  

Settings are controlled in 'tconfig_<template_name>.py'.

Usage: 
	CASA> execfile('mock_obs.py <template_name>')
"""

import os, sys
import numpy as np
import scipy.constants as sc
execfile('tconfig_'+sys.argv[-1]+'.py')


# Simulation setups
template = sys.argv[-1]
os.chdir(template_dir)


# Number of native channels needed to span the desired velocity range
nch = 2 * np.int(vspan / (sc.c * dfreq0 / restfreq)) + 1


# TOPO frequency corresponding to desired tuning velocity (center of SPW)
t0 = au.lstToUT(au.hoursToHMS(RAdeg / 15 + np.float(HA[:-1])), date)
dt = t0[0][:-3].replace('-', '/').replace(' ','/')
nu_tune = au.restToTopo(restfreq, 1e-3 * vtune, dt, RA, DEC)


# Generate a dummy cube
dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])  
res = ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)  
ia.done()


# Simulate observations to generate the MS structure
os.chdir('sims')


# Run the simulation
simobserve(project=template+'.sim', skymodel='../dummy.image',
           antennalist=simobs_dir+'alma.cycle7.'+config+'.cfg',
           totaltime=ttotal, integration=integ, thermalnoise='', 
           indirection='J2000 '+RA+' '+DEC, incell='0.02arcsec',
           incenter=str(nu_tune/1e9)+'GHz', inwidth=str(dfreq0 * 1e-3)+'kHz', 
           refdate=date, hourangle=HA, mapsize='10arcsec', outframe='TOPO')
os.system('rm -rf *.last')
os.chdir('..')


# Move the simulated MS here
os.system('rm -rf '+template+'.ms*')
sim_MS = 'sims/'+template+'.sim/'+template+'.sim.alma.cycle7.'+config+'.ms'
os.system('mv '+sim_MS+' '+template+'.ms')


# Access MS contents
tb.open(template+'.ms')
data = np.squeeze(tb.getcol("DATA"))
u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
weights = tb.getcol('WEIGHT')
times = tb.getcol("TIME")
tb.close()


# Index the timestamps
tstamps = np.unique(times)
tstamp_ID = np.empty_like(times)
for i in range(len(tstamps)):
    tstamp_ID[times == tstamps[i]] = i


# TOPO frequency channels (Hz)
tb.open(template+'.ms/SPECTRAL_WINDOW')
nu_TOPO = np.squeeze(tb.getcol('CHAN_FREQ'))
tb.close()


# LSRK frequencies (Hz) for each timestamp
nu_LSRK = np.empty((len(tstamps), len(nu_TOPO)))
ms.open(template+'.ms')
for j in range(len(tstamps)):
    nu_LSRK[j,:] = ms.cvelfreqs(mode='channel', outframe='LSRK',
                                obstime=str(tstamps[j])+'s')
ms.close()


# Record the results
np.savez_compressed(template+'.npz', data=data, um=u, vm=v, weights=weights, 
                    tstamp_ID=tstamp_ID, nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK)


# Clean up
os.system('rm -rf dummy.image')
os.chdir('..')
os.system('rm -rf *.last')
