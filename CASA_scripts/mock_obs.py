import os
import numpy as np
execfile('mconfig.py')
execfile('const.py')

# generate MS template
os.chdir('obs_templates/sims/')
simobserve(project=template+'.sim', 
           skymodel='../'+template+'.fits', 
           antennalist=simobs_dir+'alma.cycle7.'+config+'.cfg',
           totaltime=ttotal, integration=integ, 
           thermalnoise='', refdate=date, hourangle=HA, mapsize='10arcsec')

# note the simulated MS filename
sim_MS = template+'.sim/'+template+'.sim.alma.cycle7.'+config+'.ms'

# open the MS table and extract the measurement times
tb.open(sim_MS)
times = tb.getcol("TIME")
tb.close()

# open the MS table and extract the frequencies 
tb.open(sim_MS+'/SPECTRAL_WINDOW')
nchan = tb.getcol('NUM_CHAN').tolist()[0]
freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
tb.close()

# identify unique timestamps (in MJD)
tstamps = np.unique(times)

# get a date/time string corresponding to the start of the EB
datetime0 = au.mjdsecToTimerangeComponent(tstamps[0])

# set the fixed TOPO frequencies ("Doppler setting")
vLSRK_0 = 1e-3 * c * (1 - freqlist[0] / restfreq)
#print(1e-3 * c * (1 - freqlist / restfreq))
print(vLSRK_0)
freq_TOPO = au.restToTopo(restfreq, vLSRK_0, datetime0, RA, DEC) - \
            dfreq0 * np.arange(nchan)

# calculate the LSRK frequencies that correspond to these TOPO frequencies at
# each individiaul timestamp
freq_LSRK = np.empty((len(tstamps), nchan))
for i in range(len(tstamps)):
    dt = au.mjdsecToTimerangeComponent(tstamps[i])
    for j in range(nchan):
        freq_LSRK[i,j] = au.topoToLSRK(freq_TOPO[j], dt, RA, DEC)










os.chdir('../')
