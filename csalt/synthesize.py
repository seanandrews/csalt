"""
    Scripts for generating a synthetic dataset based on the inputs in a 
    configuration file.  (details below for each subroutine)
"""

import os, sys, importlib
import numpy as np
import h5py
from csalt.data import dataset
from csalt.models import vismodel_full, vismodel_def
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






def make_data(cfg_file, mtype='csalt', make_raw_FITS=True):

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

        # Load the template information into a dataset object
        tmp = h5py.File(inp.template_dir+inp.template[EB]+'.h5', "r")
        tmp_um, tmp_vm = np.asarray(tmp['um']), np.asarray(tmp['vm'])
        tmp_vis = np.asarray(tmp['vis_real'])+1.0j*np.asarray(tmp['vis_imag'])
        tmp_wgts = np.asarray(tmp['weights'])
        tmp_TOPO = np.asarray(tmp['nu_TOPO'])
        tmp_LSRK = np.asarray(tmp['nu_LSRK'])
        tmp_stmp = np.asarray(tmp['tstamp_ID'])
        tmp_dataset = dataset(tmp_um, tmp_vm, tmp_vis, tmp_wgts, 
                              tmp_TOPO, tmp_LSRK, tmp_stmp)
        tmp.close()

        # Calculate the model visibilities in pure (p) and noisy (n) cases
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist, inp.cfg_dict
        if mtype == 'csalt':
            mvis_p, mvis_n = vismodel_full(inp.pars, fixed, tmp_dataset, 
                                           oversample=inp.nover, 
                                           noise_inject=inp.RMS[EB])
        else:
            mvis_p, mvis_n = vismodel_def(inp.pars, fixed, tmp_dataset,
                                          noise_inject=inp.RMS[EB])

        # Calculate model weights
        sigma_out = 1e-3 * inp.RMS[EB] * \
                    np.sqrt(tmp_dataset.npol * tmp_dataset.nvis)
        mwgt = np.sqrt(1 / sigma_out) * np.ones_like(tmp_dataset.wgt)

        # Store "raw" synthetic data in HDF5 format (for each EB)
        # (e.g., this can have a larger vel range / no time-averaging, etc., 
        #  that we may want to have on-hand without fully re-generating it)
        hdf_out = inp.synthraw_dir+inp.basename+'/'+inp.basename+'_EB'+str(EB)
        os.system('rm -rf '+hdf_out+'.h5')
        outp = h5py.File(hdf_out+'.h5', "w")
        outp.create_dataset("um", tmp_um.shape, dtype='float64')[:] = tmp_um
        outp.create_dataset("vm", tmp_vm.shape, dtype='float64')[:] = tmp_vm
        outp.create_dataset("mvis_pure_real", mvis_p.shape, 
                            dtype="float64")[:,:,:] = mvis_p.real
        outp.create_dataset("mvis_pure_imag", mvis_p.shape, 
                            dtype="float64")[:,:,:] = mvis_p.imag
        outp.create_dataset("mvis_noisy_real", mvis_n.shape, 
                            dtype="float64")[:,:,:] = mvis_n.real
        outp.create_dataset("mvis_noisy_imag", mvis_n.shape, 
                            dtype="float64")[:,:,:] = mvis_n.imag
        outp.create_dataset("weights", mwgt.shape, dtype="float64")[:,:] = mwgt
        outp.create_dataset("nu_TOPO", tmp_TOPO.shape, 
                            dtype="float64")[:] = tmp_TOPO
        outp.create_dataset("nu_LSRK", tmp_LSRK.shape, 
                            dtype="float64")[:,:] = tmp_LSRK
        outp.create_dataset("tstamp_ID", tmp_stmp.shape, 
                            dtype="float64")[:] = tmp_stmp
        outp.close()


    # Pack these outputs into "raw", concatenated MS files (like real data)
    os.system('rm -rf '+inp.casalogs_dir+'pack_synth_data.'+cfg_file+'.log')
    os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
              'pack_synth_data.'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/pack_synth_data.py '+cfg_file)

    sys.exit()

    # Format the data into "reduced" form (+ time-average if desired) 
    os.system('rm -rf '+inp.casalogs_dir+'format_data.'+cfg_file+'.log')
    os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
              'format_data.'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/format_data.py configs/generate_'+ \
              cfg_file+' pure')
    os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
              'format_data.'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/format_data.py configs/generate_'+ \
              cfg_file+' noisy')

    return 0
