import os, sys
import numpy as np
execfile('sconfig_'+sys.argv[-1]+'.py')
execfile('const.py')

"""
Load the synthetic visibility data.
"""
# load the synthetic visibility spectra
if stordir[-1] != '/': stordir += '/'
dat = np.load(stordir+basename+'/'+basename+'-'+template+'.npz')
ivis, ivis_pure, iwgt = dat['vis'], dat['vis_pure'], dat['weights']


# open the "pure" (uncorrupted) MS table and substitute in the synthetic data
os.system('rm -rf '+stordir+basename+'/'+basename+'-'+template+'.pure.ms')
os.system('cp -r obs_templates/'+template+'.ms '+ \
          stordir+basename+'/'+basename+'-'+template+'.pure.ms')
tb.open(stordir+basename+'/'+basename+'-'+template+'.pure.ms', nomodify=False)
tb.putcol("DATA", ivis_pure)
tb.putcol("WEIGHT", iwgt)
tb.flush()
tb.close()

# open the "noisy" (corrupted) MS table and substitute in the synthetic data
os.system('rm -rf '+stordir+basename+'/'+basename+'-'+template+'.noisy.ms')
os.system('cp -r obs_templates/'+template+'.ms '+ \
          stordir+basename+'/'+basename+'-'+template+'.noisy.ms')
tb.open(stordir+basename+'/'+basename+'-'+template+'.noisy.ms', nomodify=False)
tb.putcol("DATA", ivis)
tb.putcol("WEIGHT", iwgt)
tb.flush()
tb.close()

os.system('rm -rf *.last')
