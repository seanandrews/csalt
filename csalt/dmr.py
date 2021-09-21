import os, sys, importlib
import numpy as np
from csalt.data import dataset
from csalt.models import vismodel_def as vismodel
sys.path.append('configs/')


def dmr(cfg_file):

    # Load the configuration file contents
    try:
        inp = importlib.import_module('mconfig_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('trying to use configs/mconfig_'+cfg_file+'.py\n')
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
    os.system('rm -rf '+inp.casalogs_dir+'process_dmr_'+cfg_file+'.log')
    os.system('casa --nologger --logfile '+ \
              inp.casalogs_dir+'process_dmr_'+cfg_file+'.log '+ \
              '-c csalt/CASA_scripts/process_dmr.py '+cfg_file)

    return 0


def img_cube(cfg_file, cubetype='DAT', makemask=True):

    # Load the configuration file contents
    try:
        inp = importlib.import_module('mconfig_'+cfg_file)
    except:
        print('\nThere is a problem with the configuration file:')
        print('trying to use configs/mconfig_'+cfg_file+'.py\n')
        sys.exit()


    os.system('rm -rf '+inp.casalogs_dir+'image_cube_'+cfg_file+'.'+ \
              cubetype+'.log')
    os.system('casa --nologger --logfile '+ \
              inp.casalogs_dir+'image_cube_'+cfg_file+'.'+cubetype+'.log '+ \
              '-c csalt/CASA_scripts/image_cube.py '+\
              cfg_file+' '+cubetype+' '+ \
              str(makemask))

    return 0
