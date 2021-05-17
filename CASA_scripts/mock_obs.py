"""
Generate a template MS and associated information file for use in a synthetic
data product.
"""

import os
import numpy as np
execfile('mconfig.py')
execfile('const.py')

# generate the MS structure
os.chdir('obs_templates/sims/')
simobserve(project=template+'.sim', skymodel='../'+template+'.fits', 
           antennalist=simobs_dir+'alma.cycle7.'+config+'.cfg',
           totaltime=ttotal, integration=integ, thermalnoise='', refdate=date, 
           hourangle=HA, mapsize='10arcsec')
os.chdir('../')

# grab the simulated MS, store it in obs_template with appropriate name 
sim_MS = 'sims/'+template+'.sim/'+template+'.sim.alma.cycle7.'+config+'.ms'
os.system('rm -rf '+template+'.ms*')
os.system('mv '+sim_MS+' '+template+'.ms')

# open the MS table and extract the relevant information
tb.open(template+'.ms')
data = np.squeeze(tb.getcol("DATA"))
uvw = tb.getcol("UVW")
weights = tb.getcol("WEIGHT")
times = tb.getcol("TIME")
tb.close()

# open the MS table and extract the channel frequencies 
tb.open(template+'.ms/SPECTRAL_WINDOW')
nchan = tb.getcol('NUM_CHAN').tolist()[0]
freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
tb.close()

# identify unique timestamps (in MJD)
tstamps = np.unique(times)

# get a date/time string corresponding to the start of the EB
datetime0 = au.mjdsecToTimerangeComponent(tstamps[0])

### "Doppler setting"
# set the fixed TOPO channel frequencies (at the start of the EB) 
# and their corresponding LSRK frequencies at each timestamp
freq_TOPO = np.empty(nchan)
for j in range(nchan):
    freq_TOPO[j] = au.restToTopo(restfreq, 
                                 1e-3 * c * (1 - freqlist[j] / restfreq),
                                 datetime0, RA, DEC)

freq_LSRK = np.empty((len(tstamps), nchan))
for i in range(len(tstamps)):
    datetimei = au.mjdsecToTimerangeComponent(tstamps[i])
    for j in range(nchan):
        freq_LSRK[i,j] = au.topoToLSRK(freq_TOPO[j], datetimei, RA, DEC)


# record the outputs in a more easily-accessed file (outside CASA)
np.savez(template+'.npz', data=data, uvw=uvw, weights=weights, 
                          freq_TOPO=freq_TOPO, freq_LSRK=freq_LSRK)
