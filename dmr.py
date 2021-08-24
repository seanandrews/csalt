import os, sys, importlib
import numpy as np
from csalt_data import dataset
from csalt_models import vismodel_def as vismodel
sys.path.append('configs_modeling/')


### Ingest the configuration file (in configs_modeling/). 
if len(sys.argv) == 1:
    print('\nSpecify a configuration filename in configs_modeling/ as an '+ \
          'argument: e.g., \n')
    print('        python dmr.py <config_filename>\n')
    sys.exit()
else:
    # Read user-supplied argument
    config_filename = sys.argv[1]

    # Strip the '.py' if they included it
    if config_filename[-3:] == '.py':
        config_filename = config_filename[:-3]

    # Load the configuration file contents
    try:
        inp = importlib.import_module('mconfig_'+config_filename)
    except:
        print('\nThere is a problem with the configuration file:')
        print('trying to use configs_modeling/sconfig_'+config_filename+'.py\n')
        sys.exit()


# load metadata
data_dict = np.load(inp.dataname+'.npy', allow_pickle=True).item()


# calculate
for EB in range(data_dict['nobs']):

    # set fixed parameters
    fixed = inp.nu_rest, inp.FOV[EB], inp.Npix[EB], inp.dist

    # load dataset
    d_ = np.load(inp.dataname+inp._ext+'_EB'+str(EB)+'.npz')
    vdata = dataset(d_['um'], d_['vm'], d_['data'], d_['weights'],
                    d_['nu_TOPO'], d_['nu_LSRK'], d_['tstamp_ID'])

    # calculate visibilities
    modelvis = vismodel(inp.pars, fixed, vdata)

    # pack dataset and model back into file
    os.system('rm -rf '+inp.dataname+inp._ext+'_EB'+str(EB)+'.MOD.npz')
    np.savez_compressed(inp.dataname+inp._ext+'_EB'+str(EB)+'.MOD', 
                        model=modelvis, resid=vdata.vis - modelvis)



# convert the model and residual visibilities into MS format
os.system('rm -rf CASA_logs/process_DMR_'+config_filename+'.log')
os.system('casa --nologger --logfile '+ \
          'CASA_logs/process_DMR_'+config_filename+'.log '+ \
          '-c CASA_scripts/process_DMR.py '+config_filename)
