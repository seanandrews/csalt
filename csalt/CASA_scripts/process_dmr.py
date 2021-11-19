"""
    This CASA script packs a set of model and residual visibilities -- saved in 
    an HDF5 file -- into their corresponding MS formats (for subsequent use in 
    CASA; e.g., for imaging).  It is called inside csalt.dmr() as

        casa -c process_dmr.py <cfg_file> 

    where <cfg_file> is the relevant part of the configuration input filename.
    (This *will* change when we update to full CASA v6.x.)

    This script will output ...

"""

import os, sys
import numpy as np
import h5py


# Load configuration file
cfg_file = sys.argv[-1]
if os.path.exists('configs/mdl_'+cfg_file+'.py'):
    execfile('configs/mdl_'+cfg_file+'.py')
else:
    print('Could not find input configuration file:')
    print('"configs/mdl_'+cfg_file+'.py" is not available.')
    sys.exit()

# Make sure the MODEL file exists
if not os.path.exists(dataname+_ext+'.MODEL.h5'):
    print('Could not find MODEL input HDF5 file:')
    print('"'+dataname+_ext+'.MODEL.h5" is not available.')
    sys.exit()

# Load the MODEL file
f = h5py.File(dataname+_ext+'.MODEL.h5', "r")

# Compile the sub-MS directory name
sub_dir = reduced_dir+basename+'/subMS/'

# process the model and residual MS
concat_model_files = []
concat_resid_files = []
for EB in range(f.attrs["nobs"]):
    # Load the model and residual visibilities
    mvis = np.asarray(f['EB'+str(EB)+'/model_real']) + 1.0j* \
           np.asarray(f['EB'+str(EB)+'/model_imag'])   
    rvis = np.asarray(f['EB'+str(EB)+'/resid_real']) + 1.0j* \
           np.asarray(f['EB'+str(EB)+'/resid_imag'])

    # Copy over the DATA MS file for the MODEL and RESID
    os.system('rm -rf '+sub_dir+basename+_ext+'_EB'+str(EB)+'.MODEL.ms*')
    os.system('rm -rf '+sub_dir+basename+_ext+'_EB'+str(EB)+'.RESID.ms*')
    os.system('cp -r '+sub_dir+basename+_ext+'_EB'+str(EB)+'.DATA.ms '+ \
                       sub_dir+basename+_ext+'_EB'+str(EB)+'.MODEL.ms')
    os.system('cp -r '+sub_dir+basename+_ext+'_EB'+str(EB)+'.DATA.ms '+ \
                       sub_dir+basename+_ext+'_EB'+str(EB)+'.RESID.ms')

    # Replace DATA visibilities with MODEL, RESID visibilities (as appropriate)
    tb.open(sub_dir+basename+_ext+'_EB'+str(EB)+'.MODEL.ms', nomodify=False)
    tb.putcol("DATA", mvis)
    tb.close()
    tb.open(sub_dir+basename+_ext+'_EB'+str(EB)+'.RESID.ms', nomodify=False)
    tb.putcol("DATA", rvis)
    tb.close()

    # Append filenames
    concat_model_files += [sub_dir+basename+_ext+'_EB'+str(EB)+'.MODEL.ms']
    concat_resid_files += [sub_dir+basename+_ext+'_EB'+str(EB)+'.RESID.ms']

f.close()


# Concatenate the MS files
os.system('rm -rf '+dataname+_ext+'.MODEL.ms')
os.system('rm -rf '+dataname+_ext+'.RESID.ms')
if len(concat_model_files) > 1:
    concat(vis=concat_model_files, concatvis=dataname+_ext+'.MODEL.ms', 
           dirtol='0.1arcsec', copypointing=False)
    concat(vis=concat_resid_files, concatvis=dataname+_ext+'.RESID.ms',
           dirtol='0.1arcsec', copypointing=False)
else:
    os.system('cp -r '+concat_model_files[0]+' '+dataname+_ext+'.MODEL.ms')
    os.system('cp -r '+concat_resid_files[0]+' '+dataname+_ext+'.RESID.ms')


# Cleanup
os.system('rm -rf *.last')
