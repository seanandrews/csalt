import os
import numpy as np
execfile('mconfig.py')
execfile('const.py')

# regrid (linear interpolate) onto desired output LSRK channels
os.system('rm -rf data/'+basename+'.ms*')
mstransform(vis='obs_templates/'+template+'.ms', 
            outputvis='data/'+basename+'.ms', datacolumn='data', regridms=True,
            mode='velocity', outframe='LSRK', veltype='radio',
            start=str(chanstart_out / 1e3)+'km/s', 
            width=str(chanwidth_out / 1e3)+'km/s', nchan=nchan_out,
            restfreq=str(restfreq / 1e9)+'GHz')


# open the MS table and extract the measurement times
#tb.open(template+'.ms')
#data = np.squeeze(tb.getcol("DATA"))
#uvw = tb.getcol("UVW")
#weights = tb.getcol("WEIGHT")
#times = tb.getcol("TIME")
#tb.close()

# open the MS table and extract the frequencies 
tb.open('data/'+basename+'.ms/SPECTRAL_WINDOW')
nchan = tb.getcol('NUM_CHAN').tolist()[0]
freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
tb.close()

print(freqlist)

# record outcomes
#np.savez(template+'.npz', data=data, uvw=uvw, weights=weights, 
#                          freq_TOPO=freq_TOPO, freq_LSRK=freq_LSRK)
