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

# Create output HDF5 files
os.system('rm -rf '+dataname+'.pure.h5')
f = h5py.File(dataname+'.pure.h5', "w")
f.attrs["nobs"] = len(template)
f.close()
f = h5py.File(dataname+'.noisy.h5', "w")
f.attrs["nobs"] = len(template)
f.close()

# Loop through EBs to generate individual (pure, noisy) MS files
if synthraw_dir[-1] != '/': synthraw_dir += '/'
if template_dir[-1] != '/': template_dir += '/'
pure_H5_files, noisy_H5_files = [], []
pure_MS_files, noisy_MS_files = [], []
for EB in range(len(template)):

    # Load the data for this EB from the constituent HDF5 files
    dfile = synthraw_dir+basename+'/'+basename+'_EB'+str(EB)
    dat = h5py.File(dfile+'.pure.h5', "r")
    p_um, p_vm = np.asarray(dat["um"]), np.asarray(dat["vm"])
    p_vis_real = np.asarray(dat["vis_real"])
    p_vis_imag = np.asarray(dat["vis_imag"])
    p_weights = np.asarray(dat["weights"])
    p_nu_TOPO = np.asarray(dat["nu_TOPO"])
    p_nu_LSRK = np.asarray(dat["nu_LSRK"])
    p_tstamp = np.asarray(dat["tstamp_ID"])
    dat.close()
    data_pure = p_vis_real + 1j * p_vis_imag

    dat = h5py.File(dfile+'.noisy.h5', "r")
    n_um, n_vm = np.asarray(dat["um"]), np.asarray(dat["vm"])
    n_vis_real = np.asarray(dat["vis_real"])
    n_vis_imag = np.asarray(dat["vis_imag"])
    n_weights = np.asarray(dat["weights"])
    n_nu_TOPO = np.asarray(dat["nu_TOPO"])
    n_nu_LSRK = np.asarray(dat["nu_LSRK"])
    n_tstamp = np.asarray(dat["tstamp_ID"])
    dat.close()
    data_noisy = n_vis_real + 1j * n_vis_imag

    # Pack those data back into the "reduced" output files
    f = h5py.File(dataname+'.pure.h5', "a")
    f.create_dataset('EB'+str(EB)+'/um', data=p_um)
    f.create_dataset('EB'+str(EB)+'/vm', data=p_vm)
    f.create_dataset('EB'+str(EB)+'/vis_real', data=p_vis_real)
    f.create_dataset('EB'+str(EB)+'/vis_imag', data=p_vis_imag)
    f.create_dataset('EB'+str(EB)+'/weights', data=p_weights)
    f.create_dataset('EB'+str(EB)+'/nu_TOPO', data=p_nu_TOPO)
    f.create_dataset('EB'+str(EB)+'/nu_LSRK', data=p_nu_LSRK)
    f.create_dataset('EB'+str(EB)+'/tstamp_ID', data=p_tstamp)
    f.close()

    f = h5py.File(dataname+'.noisy.h5', "a")
    f.create_dataset('EB'+str(EB)+'/um', data=n_um)
    f.create_dataset('EB'+str(EB)+'/vm', data=n_vm)
    f.create_dataset('EB'+str(EB)+'/vis_real', data=n_vis_real)
    f.create_dataset('EB'+str(EB)+'/vis_imag', data=n_vis_imag)
    f.create_dataset('EB'+str(EB)+'/weights', data=n_weights)
    f.create_dataset('EB'+str(EB)+'/nu_TOPO', data=n_nu_TOPO)
    f.create_dataset('EB'+str(EB)+'/nu_LSRK', data=n_nu_LSRK)
    f.create_dataset('EB'+str(EB)+'/tstamp_ID', data=n_tstamp)
    f.close()


    # Get the TOPO channel frequencies from the template MS
    tb.open(template_dir+template[EB]+'.ms/SPECTRAL_WINDOW')
    nu_all = np.squeeze(tb.getcol('CHAN_FREQ'))
    tb.close()

    # Identify the subset of channels available in the model
    chlo = np.argmin(np.abs(nu_all - n_nu_TOPO.min()))
    chhi = np.argmin(np.abs(nu_all - n_nu_TOPO.max()))
    if chlo < chhi:
        spwtag = '0:'+str(chlo)+'~'+str(chhi)
    else:
        spwtag = '0:'+str(chhi)+'~'+str(chlo)

    # Split the relevant channels from the blank (template) MS into 
    # pure, noisy MS files (still "empty")
    os.system('rm -rf '+dfile+'.*.ms*')
    split(vis=template_dir+template[EB]+'.ms', outputvis=dfile+'.pure.ms',
          datacolumn='data', spw=spwtag)
    os.system('cp -r '+dfile+'.pure.ms '+dfile+'.noisy.ms')

    # Pack the data into the "pure" MS table
    tb.open(dfile+'.pure.ms', nomodify=False)
    tb.putcol("DATA", data_pure)
    tb.putcol("WEIGHT", p_weights)
    tb.flush()
    tb.close()

    # Pack the data into the "noisy" MS table
    tb.open(dfile+'.noisy.ms', nomodify=False)
    tb.putcol("DATA", data_noisy)
    tb.putcol("WEIGHT", n_weights)
    tb.flush()
    tb.close()

    # Update file lists
    pure_H5_files += [dfile+'.pure.h5']
    noisy_H5_files += [dfile+'.noisy.h5']
    pure_MS_files += [dfile+'.pure.ms']
    noisy_MS_files += [dfile+'.noisy.ms']


# Concatenate the MS files into the data directory
os.system('rm -rf '+dataname+'.pure.ms*')
os.system('rm -rf '+dataname+'.noisy.ms*')
if len(template) > 1:
    concat(vis=pure_MS_files, concatvis=dataname+'.pure.ms',
           dirtol='0.1arcsec', copypointing=False)
    concat(vis=noisy_MS_files, concatvis=dataname+'.noisy.ms',
           dirtol='0.1arcsec', copypointing=False)
else:
    os.system('cp -r '+pure_MS_files[0]+' '+dataname+'.pure.ms')
    os.system('cp -r '+noisy_MS_files[0]+' '+dataname+'.noisy.ms')

# Cleanup 
for EB in range(len(pure_MS_files)):
    os.system('rm -rf '+pure_H5_files[EB])
    os.system('rm -rf '+noisy_H5_files[EB])
    os.system('rm -rf '+pure_MS_files[EB])
    os.system('rm -rf '+noisy_MS_files[EB])
os.system('rm -rf *.last')
