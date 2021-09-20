import os, sys, importlib
import numpy as np
from csalt.data_classes import dataset
from csalt.models import vismodel_full as vm_full
sys.path.append('configs_synth/')


def synth_data(config_filename):

    ### Ingest the configuration file (in configs_synth/). 
    try:
        inp = importlib.import_module('sconfig_'+config_filename)
    except:
        print('\nThere is a problem with the configuration file:') 
        print('  trying to use configs_synth/sconfig_'+config_filename+'.py\n')
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
    os.system('cp configs_synth/sconfig_'+config_filename+'.py ' + \
              inp.storage_dir+inp.basename)

    # Check and setup "reduced" data storage space (if necessary)
    if inp.reduced_dir[-1] != '/': inp.reduced_dir += '/'
    if not os.path.exists(inp.reduced_dir+inp.basename):
        os.system('mkdir '+inp.reduced_dir)
        os.system('mkdir '+inp.reduced_dir+inp.basename)


    # Loop through simulated observations (i.e., through EBs)
    for EB in range(len(inp.template)):

        # Generate the (blank) template (u,v) tracks (with CASA.simobserve)
        # (output MS files stored in obs_templates/)
        tmp_ = config_filename+'_'+str(EB)
        if EB == 0:
            os.system('rm -rf ../CASA_logs/gen_template.'+tmp_+'.log')
        os.system('casa --nologger --logfile ../CASA_logs/'+ \
                  'gen_template.'+tmp_+'.log -c ' + \
                  'CASA_scripts/gen_template.py '+config_filename+' '+str(EB))


        # Load the template information into a dataset object
        tmp = np.load(inp.template_dir+inp.template[EB]+'.npz')
        tmp_dataset = dataset(tmp['um'], tmp['vm'], tmp['data'], 
                              tmp['weights'], tmp['nu_TOPO'], tmp['nu_LSRK'], 
                              tmp['tstamp_ID'])


        # Calculate model visibilities
        fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist
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
    os.system('rm -rf ../CASA_logs/pack_synth_data.'+config_filename+'.log')
    os.system('casa --nologger --logfile ../CASA_logs/'+ \
              'pack_synth_data.'+config_filename+'.log '+ \
              '-c CASA_scripts/pack_synth_data.py '+config_filename)



    # Format the data (+ time-average if desired) 
    os.system('rm -rf ../CASA_logs/format_data.'+config_filename+'.log')
    os.system('casa --nologger --logfile ../CASA_logs/'+ \
              'format_data.'+config_filename+'.log '+ \
              '-c CASA_scripts/format_data.py configs_synth/sconfig_'+ \
              config_filename+' pure')
    os.system('casa --nologger --logfile ../CASA_logs/'+ \
              'format_data.'+config_filename+'.log '+ \
              '-c CASA_scripts/format_data.py configs_synth/sconfig_'+ \
              config_filename+' noisy')

    return 0
