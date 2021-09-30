import os, sys, importlib
import numpy as np
from csalt.data import dataset
from csalt.models import vismodel_full as vm_full
sys.path.append('configs/')


def make_data(cfg_file):

    ### Ingest the configuration file (in configs/). 
    try:
        inp = importlib.import_module('generate_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:') 
        print('  trying to use configs/generate_'+cfg_file+'.py\n')
        sys.exit()


    ### Check workspace and set up properly if necessary.
    # Check and setup template / simulation space (if necessary)
    if inp.template_dir[-1] != '/': inp.template_dir += '/'
    if not os.path.exists(inp.template_dir):
        os.system('mkdir '+inp.template_dir)
        os.system('mkdir '+inp.template_dir+'sims')
    elif not os.path.exists(inp.template_dir+'sims'):
        os.system('mkdir '+inp.template_dir+'sims')

    # Check and setup "raw" data storage space (if necessary)
    if inp.storage_dir[-1] != '/': inp.storage_dir += '/'
    if not os.path.exists(inp.storage_dir+inp.basename):
        os.system('mkdir '+inp.storage_dir)
        os.system('mkdir '+inp.storage_dir+inp.basename)
    os.system('cp configs/generate_'+cfg_file+'.py ' + \
              inp.storage_dir+inp.basename)

    # Check and setup "reduced" data storage space (if necessary)
    if inp.reduced_dir[-1] != '/': inp.reduced_dir += '/'
    if not os.path.exists(inp.reduced_dir+inp.basename):
        os.system('mkdir '+inp.reduced_dir)
        os.system('mkdir '+inp.reduced_dir+inp.basename)

    # Check and setup CASA logs repository (if necessary)
    if inp.casalogs_dir[-1] != '/': inp.casalogs_dir += '/'
    if not os.path.exists(inp.casalogs_dir):
        os.system('mkdir '+inp.casalogs_dir)


    # Loop through simulated observations (i.e., through EBs)
    for EB in range(len(inp.template)):

        # Generate the (blank) template (u,v) tracks (with CASA.simobserve)
        # (output MS files stored in obs_templates/)
        tmp_ = cfg_file+'_'+str(EB)
        if EB == 0:
            os.system('rm -rf '+inp.casalogs_dir+'/gen_template.'+tmp_+'.log')
        os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
                  'gen_template.'+tmp_+'.log -c '+ \
                  'csalt/CASA_scripts/gen_template.py '+cfg_file+' '+str(EB))


        # Load the template information into a dataset object
        tmp = np.load(inp.template_dir+inp.template[EB]+'.npz')
        tmp_dataset = dataset(tmp['um'], tmp['vm'], tmp['data'], 
                              tmp['weights'], tmp['nu_TOPO'], tmp['nu_LSRK'], 
                              tmp['tstamp_ID'])


        # Calculate model visibilities
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist, inp.cfg_dict
        mvis_p, mvis_n = vm_full(inp.pars, fixed, tmp_dataset, 
                                 oversample=inp.nover, noise_inject=inp.RMS[EB])


        # Calculate model weights
        sigma_out = 1e-3 * inp.RMS[EB] * \
                    np.sqrt(tmp_dataset.npol * tmp_dataset.nvis)
        mwgt = np.sqrt(1 / sigma_out) * np.ones_like(tmp_dataset.wgt)


        # Store synthetic data in .npz format (for each sim / EB)
        npz_out = inp.storage_dir+inp.basename+'/'+inp.basename+'_EB'+str(EB)
        os.system('rm -rf '+npz_out+'.npz')
        np.savez_compressed(npz_out+'.npz', u=tmp['um'], v=tmp['vm'], 
                            data_pure=mvis_p, data_noisy=mvis_n, weights=mwgt, 
                            nu_TOPO=tmp['nu_TOPO'], nu_LSRK=tmp['nu_LSRK'],
                            tstamp_ID=tmp['tstamp_ID'])



    # Pack the data into a single, concatenated MS file (like real data)
    os.system('rm -rf '+inp.casalogs_dir+'pack_synth_data.'+cfg_file+'.log')
    os.system('casa --nologger --logfile '+inp.casalogs_dir+ \
              'pack_synth_data.'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/pack_synth_data.py '+cfg_file)



    # Format the data (+ time-average if desired) 
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
