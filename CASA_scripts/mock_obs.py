"""
Generate a template MS and associated information file for use in a synthetic
data product.
"""
import os, sys
import numpy as np
execfile('sconfig_'+sys.argv[-1]+'.py')
execfile('const.py')


### Simulation setups
os.chdir('obs_templates')

# Number of channels needed to span the desired velocity range
nch = 2 * np.int(vspan / (c_ * dfreq0 / restfreq)) + 1

# TOPO frequency corresponding to desired tuning velocity (center of SPW)
t0 = au.lstToUT(au.hoursToHMS(RAdeg / 15 + np.float(HA[:-1])), date)
dt = t0[0][:-3].replace('-', '/').replace(' ','/')
nu_tune = au.restToTopo(restfreq, 1e-3 * vtune, dt, RA, DEC)

# Generate a dummy cube
dummy = ia.makearray(v=0.001, shape=[64, 64, 4, nch])  
res = ia.fromarray(outfile='dummy.image', pixels=dummy, overwrite=True)  
ia.done()



### Simulate observations to generate the MS structure
os.chdir('sims')

# Run the simulation
simobserve(project=template+'.sim', skymodel='../dummy.image',
           antennalist=simobs_dir+'alma.cycle7.'+config+'.cfg',
           totaltime=ttotal, integration=integ, thermalnoise='', 
           indirection='J2000 '+RA+' '+DEC, incell='0.02arcsec',
           incenter=str(nu_tune/1e9)+'GHz', inwidth=str(dfreq0 * 1e-3)+'kHz', 
           refdate=date, hourangle=HA, mapsize='10arcsec', outframe='TOPO')
os.chdir('../')

# Store the simulated MS in obs_template
os.system('rm -rf '+template+'.ms*')
sim_MS = 'sims/'+template+'.sim/'+template+'.sim.alma.cycle7.'+config+'.ms'
os.system('mv '+sim_MS+' '+template+'.ms')



### Extract the MS contents for easier access

# Data structures and unique timestamps (s in UTC MJD) 
tb.open(template+'.ms')
data = np.squeeze(tb.getcol("DATA"))
u, v = tb.getcol('UVW')[0,:], tb.getcol('UVW')[1,:]
weights = tb.getcol('WEIGHT')
tstamps = np.unique(tb.getcol("TIME"))
tb.close()

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
np.savez(template+'.npz', data=data, u=u, v=v, weights=weights, 
                          nu_TOPO=nu_TOPO, nu_LSRK=nu_LSRK)

os.chdir('../')

os.system('rm -rf *.last')
