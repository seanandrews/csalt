import os, sys
import numpy as np

mdlbase = sys.argv[-1]
execfile('sconfig_'+mdlbase+'.py')


# Loop through constituent datasets
if storage_dir[-1] != '/': storage_dir += '/'
if template_dir[-1] != '/': template_dir += '/'
pure_files, noisy_files = [], []
for itmp in range(len(template)):

    # Load the synthetic dataset
    dat = np.load(storage_dir+basename+'/'+mdlbase+'_EB'+str(itmp)+'.npz')
    data_pure, data_noisy = dat['data_pure'], dat['data_noisy']
    weights = dat['weights']

    # Copy the corresponding template MS files
    _MS = storage_dir+basename+'/'+mdlbase+'_EB'+str(itmp)
    os.system('rm -rf '+_MS+'.*.ms')
    os.system('cp -r '+template_dir+template[itmp]+'.ms '+_MS+'.pure.ms')
    os.system('cp -r '+template_dir+template[itmp]+'.ms '+_MS+'.noisy.ms')

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
os.system('rm -rf '+storage_dir+basename+'/'+mdlbase+'.*.ms')
if len(template) > 1:
    concat(vis=pure_files, 
           concatvis=storage_dir+basename+'/'+mdlbase+'.pure.ms',
           dirtol='0.1arcsec', copypointing=False)
    concat(vis=noisy_files,
           concatvis=storage_dir+basename+'/'+mdlbase+'.noisy.ms',
           dirtol='0.1arcsec', copypointing=False)
else:
    os.system('cp -r '+pure_files[0]+' '+storage_dir+basename+'/'+\
              mdlbase+'.pure.ms')
    os.system('cp -r '+noisy_files[0]+' '+storage_dir+basename+'/'+\
              mdlbase+'.noisy.ms')


# Cleanup 
for i in range(len(pure_files)):
    os.system('rm -rf '+pure_files[i])
    os.system('rm -rf '+noisy_files[i])
os.system('rm -rf '+storage_dir+basename+'/'+mdlbase+'_EB*npz')
os.system('rm -rf *.last')
