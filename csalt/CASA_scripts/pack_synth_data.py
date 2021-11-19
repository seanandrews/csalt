"""
    This CASA script takes the "raw" synthetic data stored in individual HDF5 
    files (one for each EB) and packs them into two concatenated MSs (one for 
    the "pure" [noise-free] case, and one for the "noisy" case).  You would 
    probably never execute it outside of csalt.synthesize.make_data(), but if 
    did it would be done as

        execfile('pack_synth_data.py <cfg_file>')

    where <cfg_file> is the relevant part of the configuration input filename 
    (i.e., configs/gen_<cfg_file>.py).

    The script will output ...

"""
import os, sys
import numpy as np
import h5py


"""
    Load information and prepare for reading / packaging
"""
# Parse the argument
cfg_file = sys.argv[-1]

# Load the configuration file
execfile('configs/gen_'+cfg_file+'.py')

# Loop through EBs to generate individual (pure, noisy) MS files
if synthraw_dir[-1] != '/': synthraw_dir += '/'
if template_dir[-1] != '/': template_dir += '/'
pure_files, noisy_files = [], []
for EB in range(len(template)):

    # Load the data for this EB
    dfile = synthraw_dir+basename+'/'+basename+'_EB'+str(EB)

    dat = h5py.File(dfile+'.pure.h5', "r")
    data_pure = np.asarray(dat["vis_real"]) + 1j * np.asarray(dat["vis_imag"])
    dat.close()

    dat = h5py.File(dfile+'.noisy.h5', "r")
    data_noisy = np.asarray(dat["vis_real"]) + 1j * np.asarray(dat["vis_imag"])
    weights = np.asarray(dat["weights"])
    dat.close()

    # Copy the blank (template) MS into pure, noisy MS duplicates
    _MS = synthraw_dir+basename+'/'+basename+'_EB'+str(EB)
    os.system('rm -rf '+_MS+'.*.ms*')
    os.system('cp -r '+template_dir+template[EB]+'.ms '+_MS+'.pure.ms')
    os.system('cp -r '+template_dir+template[EB]+'.ms '+_MS+'.noisy.ms')

    # Pack the data into the "pure" MS table
    tb.open(_MS+'.pure.ms', nomodify=False)
    tb.putcol("DATA", data_pure)
    tb.putcol("WEIGHT", weights)
    tb.flush()
    tb.close()

    # Pack the data into the "noisy" MS table
    tb.open(_MS+'.noisy.ms', nomodify=False)
    tb.putcol("DATA", data_noisy)
    tb.putcol("WEIGHT", weights)
    tb.flush()
    tb.close()

    # Update file lists
    pure_files += [_MS+'.pure.ms']
    noisy_files += [_MS+'.noisy.ms']


# Concatenate the MS files
os.system('rm -rf '+synthraw_dir+basename+'/'+basename+'_pure.ms*')
os.system('rm -rf '+synthraw_dir+basename+'/'+basename+'_noisy.ms*')
if len(template) > 1:
    concat(vis=pure_files, 
           concatvis=synthraw_dir+basename+'/'+cfg_file+'_pure.ms',
           dirtol='0.1arcsec', copypointing=False)
    concat(vis=noisy_files,
           concatvis=synthraw_dir+basename+'/'+cfg_file+'_noisy.ms',
           dirtol='0.1arcsec', copypointing=False)
else:
    os.system('cp -r '+pure_files[0]+' '+synthraw_dir+basename+'/'+ \
              cfg_file+'_pure.ms')
    os.system('cp -r '+noisy_files[0]+' '+synthraw_dir+basename+'/'+ \
              cfg_file+'_noisy.ms')

# Cleanup 
for EB in range(len(pure_files)):
    os.system('rm -rf '+pure_files[EB])
    os.system('rm -rf '+noisy_files[EB])
    #os.system('rm -rf '+template_dir+template[EB]+'.ms*')
    #os.system('rm -rf '+template_dir+template[EB]+'.h5')
os.system('rm -rf *.last')
