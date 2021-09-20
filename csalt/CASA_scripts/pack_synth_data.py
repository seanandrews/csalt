import os, sys
import numpy as np


# Load configuration file
fname = sys.argv[-1]
execfile('configs_synth/sconfig_'+fname+'.py')


# Loop through EBs to generate individual (pure, noisy) MS files
if storage_dir[-1] != '/': storage_dir += '/'
if template_dir[-1] != '/': template_dir += '/'
pure_files, noisy_files = [], []
for EB in range(len(template)):

    # Load the data for this EB
    dat = np.load(storage_dir+basename+'/'+basename+'_EB'+str(EB)+'.npz')
    data_pure, data_noisy = dat['data_pure'], dat['data_noisy']
    weights = dat['weights']

    # Copy the MS file into pure, noisy MS copies (still "blank")
    _MS = storage_dir+basename+'/'+basename+'_EB'+str(EB)
    os.system('rm -rf '+_MS+'.*.ms*')
    os.system('cp -r '+template_dir+template[EB]+'.ms '+_MS+'.pure.ms')
    os.system('cp -r '+template_dir+template[EB]+'.ms '+_MS+'.noisy.ms')

    # Pack the "pure" MS table
    tb.open(_MS+'.pure.ms', nomodify=False)
    tb.putcol("DATA", data_pure)
    tb.putcol("WEIGHT", weights)
    tb.flush()
    tb.close()

    # Pack the "noisy" MS table
    tb.open(_MS+'.noisy.ms', nomodify=False)
    tb.putcol("DATA", data_noisy)
    tb.putcol("WEIGHT", weights)
    tb.flush()
    tb.close()

    # Update file lists
    pure_files += [_MS+'.pure.ms']
    noisy_files += [_MS+'.noisy.ms']


# Concatenate MS files
os.system('rm -rf '+storage_dir+basename+'/'+basename+'_pure.ms*')
os.system('rm -rf '+storage_dir+basename+'/'+basename+'_noisy.ms*')
if len(template) > 1:
    concat(vis=pure_files, 
           concatvis=storage_dir+basename+'/'+fname+'_pure.ms',
           dirtol='0.1arcsec', copypointing=False)
    concat(vis=noisy_files,
           concatvis=storage_dir+basename+'/'+fname+'_noisy.ms',
           dirtol='0.1arcsec', copypointing=False)
else:
    os.system('cp -r '+pure_files[0]+' '+storage_dir+basename+'/'+ \
              fname+'_pure.ms')
    os.system('cp -r '+noisy_files[0]+' '+storage_dir+basename+'/'+ \
              fname+'_noisy.ms')


# Cleanup 
for EB in range(len(pure_files)):
    os.system('rm -rf '+pure_files[EB])
    os.system('rm -rf '+noisy_files[EB])
os.system('rm -rf '+storage_dir+basename+'/'+basename+'_EB*npz')
os.system('rm -rf *.last')
