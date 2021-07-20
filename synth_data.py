"""
    synth_data.py

    Usage: 
        > python synth_data.py <basename>

    Settings are edited by the user in 'sconfig_<basename>.py'.

    Outputs:
        - 
"""
import os, sys, importlib
import numpy as np
from classes import dataset
from csalt_models import vismodel_full as vismodel


# Load user-defined inputs
inp = importlib.import_module('sconfig_'+sys.argv[-1])


# Prepare storage space
if inp.storage_dir[-1] != '/': inp.storage_dir += '/'
if not os.path.exists(inp.storage_dir+inp.basename):
    os.system('mkdir '+inp.storage_dir)
    os.system('mkdir '+inp.storage_dir+inp.basename)
os.system('cp sconfig_'+sys.argv[-1]+'.py '+inp.storage_dir+inp.basename)


# Start loop over templates
if inp.template_dir[-1] != '/': inp.template_dir += '/'
for itmp in range(len(inp.template)):

    # Load the template information into a dataset object
    tmp = np.load(inp.template_dir+inp.template[itmp]+'.npz')
    tmp_dataset = dataset(tmp['um'], tmp['vm'], tmp['data'], tmp['weights'], 
                          tmp['nu_TOPO'], tmp['nu_LSRK'], tmp['tstamp_ID'])

    # Calculate model visibilities
    mvis_p, mvis_n = vismodel(inp.pars, inp.fixed, tmp_dataset, 
                              oversample=inp.spec_over, 
                              noise_inject=inp.RMS[itmp])

    # Calculate model weights
    sigma_out = 1e-3 * inp.RMS[itmp] * \
                np.sqrt(tmp_dataset.npol*tmp_dataset.nvis)
    mwgt = np.sqrt(1 / sigma_out) * np.ones_like(tmp_dataset.wgt)


    # Temporarily store data in .npz format for each EB
    npz_out = inp.storage_dir+inp.basename+'/'+sys.argv[-1]+'_EB'+str(itmp)
    os.system('rm -rf '+npz_out+'.npz')
    np.savez_compressed(npz_out+'.npz', u=tmp['um'], v=tmp['vm'], 
                        data_pure=mvis_p, data_noisy=mvis_n, weights=mwgt, 
                        nu_TOPO=tmp['nu_TOPO'], nu_LSRK=tmp['nu_LSRK'],
                        tstamp_ID=tmp['tstamp_ID'])


# Pack the data into a single, concatenated MS file (like real data)
os.system('rm -rf CASA_logs/pack_synth_data_'+sys.argv[-1]+'.log')
os.system('casa --nologger --logfile CASA_logs/pack_synth_data_'+sys.argv[-1]+\
          '.log -c CASA_scripts/pack_synth_data.py '+sys.argv[-1])
