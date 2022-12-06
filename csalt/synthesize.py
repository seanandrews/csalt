"""
    Scripts for generating a synthetic dataset based on the inputs in a 
    configuration file.  (details below for each subroutine)
"""

import os, sys, importlib
import numpy as np
import h5py
import copy
import matplotlib.pyplot as plt
from csalt.data import dataset, HDF_to_dataset, dataset_to_HDF
from csalt.models import vismodel_full, vismodel_def, vismodel_naif_wdoppcorr, vismodel_FITS
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
        inp = importlib.import_module('gen_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('  trying to use configs/gen_'+cfg_file+'.py\n')
        sys.exit()

    # If not available, set up the CASA/simobserve template space
    if inp.template_dir[-1] != '/': inp.template_dir += '/'
    if not os.path.exists(inp.template_dir):
        os.system('mkdir '+inp.template_dir)
        os.system('mkdir '+inp.template_dir+'sims')
    elif not os.path.exists(inp.template_dir+'sims'):
        os.system('mkdir '+inp.template_dir+'sims')

    # If not available, set up the "raw" synthetic data storage space 
    if inp.synthraw_dir[-1] != '/': inp.synthraw_dir += '/'
    if not os.path.exists(inp.synthraw_dir):
        os.system('mkdir '+inp.synthraw_dir)
        os.system('mkdir '+inp.synthraw_dir+inp.basename)
    elif not os.path.exists(inp.synthraw_dir+inp.basename):
        os.system('mkdir '+inp.synthraw_dir+inp.basename)
    os.system('cp configs/gen_'+cfg_file+'.py '+inp.synthraw_dir+inp.basename)

    # If not available, set up the "reduced" data storage space
    if inp.reduced_dir[-1] != '/': inp.reduced_dir += '/'
    if not os.path.exists(inp.reduced_dir):
        os.system('mkdir '+inp.reduced_dir)
        os.system('mkdir '+inp.reduced_dir+inp.basename)
    elif not os.path.exists(inp.reduced_dir+inp.basename):
        os.system('mkdir '+inp.reduced_dir+inp.basename)

    # If not available, set up the CASA logs repository
    if inp.casalogs_dir[-1] != '/': inp.casalogs_dir += '/'
    if not os.path.exists(inp.casalogs_dir):
        os.system('mkdir '+inp.casalogs_dir)

    # Make sure the specified antenna configuration files are available
    if inp.antcfg_dir[-1] != '/': inp.antcfg_dir += '/'
    if not os.path.exists(inp.antcfg_dir):
        print('The path to the antenna configuration files does not exist.')
        sys.exit()
    else:
        for i in range(len(inp.config)):
            if not os.path.exists(inp.antcfg_dir+inp.config[i]+'.cfg'):
                print('Cannot find antenna config file = "%s"' % inp.config[i])
                sys.exit()

    # If various things should be nEB-dimensional lists but are not, force them 
    # to have the same values in all indices...

    # If you've made it through, pass the inputs object
    return inp




def make_template(cfg_file):

    # Run checks on the setups and load the configuration file inputs
    inp = check_setups(cfg_file)

    # Loop over the desired Execution Blocks (EBs)
    for EB in range(len(inp.template)):
        # Generate (blank) template (u,v) tracks (with CASA.simobserve)
        tmp_ = cfg_file+'_'+str(EB)
        if EB == 0:
            os.system('rm -rf '+inp.casalogs_dir+'/gen_template.'+tmp_+'.log')
        os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
                  'gen_template.'+tmp_+'.log -c '+ \
                  'csalt/CASA_scripts/gen_template.py '+cfg_file+' '+str(EB))

    return




def make_data(cfg_file, mtype='CSALT', calctype='full', new_template=True):

    # Run checks on the setups and load the configuration file inputs
    inp = check_setups(cfg_file)

    # Copy as appropriate the corresponding parametric_disk script
    if not os.path.exists('parametric_disk_'+mtype+'.py'):
        print('There is no such file "parametric_disk_"+mtype+".py".\n')
        return

    # Make the (blank) template observations (if requested or necessary)
    tfiles = [inp.template_dir+i+'.h5' for i in inp.template]
    if np.logical_or(new_template,
                     np.all([os.path.exists(f) for f in tfiles]) == False):
        make_template(cfg_file)


    # Make data in loop over Execution Blocks (EBs)
    for EB in range(len(inp.template)):

        # Load the template from HDF5 into a dataset object
        tmp_dataset = HDF_to_dataset(inp.template_dir+inp.template[EB])

        # Calculate the model visibilities in pure (p) and noisy (n) cases
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist, inp.cfg_dict
        print('\n...Computing model visibilities for EB '+\
              str(EB+1)+'/'+str(len(inp.template))+'...')
        if mtype == 'FITS':
            mvis_p, mvis_n, dset_ = vismodel_FITS(inp.pars, fixed, tmp_dataset,
                                                  noise_inject=inp.RMS[EB])
            tmp_dataset = dset_
        else:
            if calctype == 'full':
                mvis_p, mvis_n = vismodel_full(inp.pars, fixed, tmp_dataset, 
                                               oversample=inp.nover, 
                                               mtype=mtype,
                                               noise_inject=inp.RMS[EB])
            else: 
                if np.logical_and(mtype == 'RADMC3D', EB > 0):
                    redo_RTimage = False
                else:
                    redo_RTimage = True
                mvis_p, mvis_n = vismodel_def(inp.pars, fixed, tmp_dataset,
                                              mtype=mtype,
                                              redo_RTimage=redo_RTimage,
                                              noise_inject=inp.RMS[EB])

        # Calculate model weights
        sigma_out = 1e-3 * inp.RMS[EB] * \
                    np.sqrt(tmp_dataset.npol * tmp_dataset.nvis)
        mwgt = np.sqrt(1 / sigma_out) * np.ones_like(tmp_dataset.wgt)

        # Populate "pure" and "noisy" datasets
        pure_dataset = copy.deepcopy(tmp_dataset)
        noisy_dataset = copy.deepcopy(tmp_dataset)
        pure_dataset.vis, noisy_dataset.vis = mvis_p, mvis_n
        pure_dataset.wgt, noisy_dataset.wgt = mwgt, mwgt

        # Store "raw" synthetic data ("pure" and "noisy") in HDF5 
        print('Writing out pure and noisy data')
        hdf_out = inp.synthraw_dir+inp.basename+'/'+inp.basename+'_EB'+str(EB)
        os.system('rm -rf '+hdf_out+'.*.h5')
        dataset_to_HDF(pure_dataset, hdf_out+'.pure')
        dataset_to_HDF(noisy_dataset, hdf_out+'.noisy')



    # Pack these outputs into "raw", concatenated MS files (like real data)
    print('\n\n...Packing model visibilities into MS files...\n\n')
    os.system('rm -rf '+inp.casalogs_dir+'pack_synth_data.'+cfg_file+'.log')
    if mtype == 'FITS':
        os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
                  'pack_synth_data.'+cfg_file+'.log '+ \
                  '-c csalt/CASA_scripts/pack_synth_data_FITS.py '+cfg_file)
    else:
        os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
                  'pack_synth_data.'+cfg_file+'.log '+ \
                  '-c csalt/CASA_scripts/pack_synth_data.py '+cfg_file)

    # Format the data into "reduced" form (+ time-average if desired) 
    if mtype != 'FITS':
        print('...Formatting MS data into reduced form...')
        os.system('rm -rf '+inp.casalogs_dir+'format_data.'+cfg_file+'.log')
        os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
                  'format_data.'+cfg_file+'.log '+ \
                  '-c csalt/CASA_scripts/format_data.py configs/gen_'+ \
                  cfg_file+' pure')
        os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
                  'format_data.'+cfg_file+'.log '+ \
                  '-c csalt/CASA_scripts/format_data.py configs/gen_'+ \
                  cfg_file+' noisy')

    return 
