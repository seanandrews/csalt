import os
import numpy as np
execfile('synth_config.py')
execfile('const.py')

"""
Load the synthetic visibility data.
"""
# load the synthetic visibility spectra
dat = np.load('storage/'+basename+'/'+basename+'-'+template+'.npz')
ivis, ivis_noisy, iwgt = dat['vis'], dat['vis_noisy'], dat['weights']


# open the "clean" (uncorrupted) MS table and substitute in the synthetic data
os.system('rm -rf storage/'+basename+'/'+basename+'-'+template+'.ms')
os.system('cp -r obs_templates/'+template+'.ms '+ \
          'storage/'+basename+'/'+basename+'-'+template+'.ms')
tb.open('storage/'+basename+'/'+basename+'-'+template+'.ms', nomodify=False)
tb.putcol("DATA", ivis)
tb.putcol("WEIGHT", iwgt)
tb.flush()
tb.close()

# open the "noisy" (corrupted) MS table and substitute in the synthetic data
os.system('rm -rf storage/'+basename+'/'+basename+'-'+template+'_noisy.ms')
os.system('cp -r obs_templates/'+template+'.ms '+ \
          'storage/'+basename+'/'+basename+'-'+template+'_noisy.ms')
tb.open('storage/'+basename+'/'+basename+'-'+template+'_noisy.ms', 
        nomodify=False)
tb.putcol("DATA", ivis_noisy)
tb.putcol("WEIGHT", iwgt)
tb.flush()
tb.close()
