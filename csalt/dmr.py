"""
    Scripts for computing (and post-processing) a single model onto an existing
    dataset, based on the inputs in a configuration file.
"""

import os, sys, importlib
import numpy as np
import h5py
from csalt.data import dataset
from csalt.models import vismodel_def as vismodel
sys.path.append('configs/')


"""
    Subroutine to check for organizational setups.  This ensures that the data 
    working directory structure is in place, that necessary ancillary files are 
    available, and that the inputs in the configuration file make sense.

    The input parameter is the name (string) of the configuration file, and the 
    subroutine (if the setup tests pass) will return an object that contains 
    the parameters defined in the configuration file.
"""
def check_setups(cfg_file):

    # Ingest the configuration file (in configs/). 
    if cfg_file[-3:] == '.py': cfg_file = cfg_file[:-3]
    try:
        inp = importlib.import_module('mdl_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('  trying to use configs/mdl_'+cfg_file+'.py\n')
        sys.exit()

    # Make sure the datafiles exist
    if not os.path.exists(inp.dataname+inp._ext+'.DATA.h5'):
        print('\nThe HDF5 file containing the base data is not found:')
        print('  "'+inp.dataname+inp._ext+'.h5" is unavailable.\n')
        sys.exit()
    if not os.path.exists(inp.dataname+inp._ext+'.DATA.ms'):
        print('\nThe MS file containing the base data is not found:')
        print('  "'+inp.dataname+inp._ext+'.ms" is unavailable.\n')
        sys.exit()

    # If not available, set up the CASA logs repository
    if inp.casalogs_dir[-1] != '/': inp.casalogs_dir += '/'
    if not os.path.exists(inp.casalogs_dir):
        os.system('mkdir '+inp.casalogs_dir)

    # If not available, set up the imaging subdirectory 
    if inp.img_dir[-1] != '/': inp.img_dir += '/'
    if not os.path.exists(inp.img_dir):
        os.system('mkdir '+inp.img_dir)

    # If various things should be nEB-dimensional lists but are not, force them 
    # to have the same values in all indices...

    # If you've made it through, pass the inputs object
    return inp



"""
    Subroutine that computes a set of model and residual spectral visibilities 
    and samples them on the frequency channels and Fourier spatial frequencies 
    of a (user-provided) dataset.

"""
def dmr(cfg_file, mtype='csalt', make_raw_FITS=True):

    # Run checks on the setups and load the configuration file inputs
    inp = check_setups(cfg_file)

    # load the data from HDF5 file
    f = h5py.File(inp.dataname+inp._ext+'.DATA.h5', "r")

    # create an output HDF5 file and assign some basic attributes
    os.system('rm -rf '+inp.dataname+inp._ext+'.MODEL.h5')
    fo = h5py.File(inp.dataname+inp._ext+'.MODEL.h5', "w")
    fo.attrs["nobs"] = f.attrs["nobs"]
    fo.attrs["datafile"] = inp.dataname+inp._ext+'.DATA.h5'
    fo.close()

    # calculate
    for EB in range(f.attrs["nobs"]):
        # set fixed parameters
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist, inp.cfg_dict

        # load data into dataset object
        d_um = np.asarray(f['EB'+str(EB)+'/um'])
        d_vm = np.asarray(f['EB'+str(EB)+'/vm'])
        d_vis = np.asarray(f['EB'+str(EB)+'/vis_real']) + 1.0j * \
                np.asarray(f['EB'+str(EB)+'/vis_imag'])
        d_wgt = np.asarray(f['EB'+str(EB)+'/weights'])
        d_TOPO = np.asarray(f['EB'+str(EB)+'/nu_TOPO'])
        d_LSRK = np.asarray(f['EB'+str(EB)+'/nu_LSRK'])
        d_stmp = np.asarray(f['EB'+str(EB)+'/tstamp_ID'])
        vdata = dataset(d_um, d_vm, d_vis, d_wgt, d_TOPO, d_LSRK, d_stmp)

        # calculate visibilities
        mvis = vismodel(inp.pars, fixed, vdata)

        # pack the model and residuals into an HDF5 output
        fo = h5py.File(inp.dataname+inp._ext+'.MODEL.h5', "a")
        fo.create_dataset("EB"+str(EB)+"/model_real", 
                          mvis.shape)[:,:,:] = mvis.real
        fo.create_dataset("EB"+str(EB)+"/model_imag",
                          mvis.shape)[:,:,:] = mvis.imag
        fo.create_dataset("EB"+str(EB)+"/resid_real",
                          mvis.shape)[:,:,:] = d_vis.real - mvis.real
        fo.create_dataset("EB"+str(EB)+"/resid_imag",
                          mvis.shape)[:,:,:] = d_vis.imag - mvis.imag
        fo.close()

    f.close()

    # convert the model and residual visibilities into MS format
    os.system('rm -rf '+inp.casalogs_dir+'process_dmr_'+cfg_file+'.log')
    os.system('casa --nologger --logfile '+ \
              inp.casalogs_dir+'process_dmr_'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/process_dmr.py '+cfg_file)

    return



"""
    Subroutine that images a specified MS.
"""
def img_cube(cfg_file, cubetype='DATA', mask_name=None):

    # Run checks on the setups and load the configuration file inputs
    inp = check_setups(cfg_file)

    # Run the imaging script
    os.system('rm -rf '+inp.casalogs_dir+'image_cube_'+cfg_file+'.'+ \
              cubetype+'.log')
    os.system('casa --nologger --logfile '+ \
              inp.casalogs_dir+'image_cube_'+cfg_file+'.'+cubetype+'.log '+ \
              '-c csalt/CASA_scripts/image_cube.py '+ \
              cfg_file+' '+cubetype+' '+ str(mask_name))

    return 
