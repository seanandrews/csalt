import os
import numpy as np
execfile('mconfig.py')
execfile('const.py')

"""
The first step is to spectrally regrid the template MS onto the desired output
LSRK channels for formatting purposes.  
"""
# regrid (linear interpolation) onto desired output LSRK channels
os.system('rm -rf data/'+basename+'-'+template+'.ms*')
mstransform(vis='obs_templates/'+template+'.ms', 
            outputvis='data/'+basename+'-'+template+'.ms', 
            datacolumn='data', regridms=True, mode='velocity', 
            outframe='LSRK', veltype='radio', nchan=nchan_out,
            start=str(chanstart_out / 1e3)+'km/s', 
            width=str(chanwidth_out / 1e3)+'km/s', 
            restfreq=str(restfreq / 1e9)+'GHz')

# copy this to a noisy MS file
os.system('rm -rf data/'+basename+'-'+template+'_noisy.ms*')
os.system('cp -r data/'+basename+'-'+template+'.ms ' + \
          'data/'+basename+'-'+template+'_noisy.ms')


"""
Now we load and pack the synthetic visibility spectra into these MS files.
"""
# load the synthetic visibility spectra
dat = np.load('data/'+basename+'-'+template+'.npz')
ifreq, iwgt = dat['freq'], dat['weights']
ivis, ivis_noisy = dat['vis'], dat['vis_noisy']

# open the "clean" (uncorrupted) MS table and substitute in the synthetic data
tb.open('data/'+basename+'-'+template+'.ms', nomodify=False)
tb.putcol("DATA", ivis)
tb.putcol("WEIGHT", iwgt)
tb.flush()
tb.close()

# open the "noisy" (corrupted) MS table and substitute in the synthetic data
tb.open('data/'+basename+'-'+template+'_noisy.ms', nomodify=False)
tb.putcol("DATA", ivis_noisy)
tb.putcol("WEIGHT", iwgt)
tb.flush()
tb.close()
