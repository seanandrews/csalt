import os, sys, importlib, time
sys.path.append('configs/')
import numpy as np
from csalt.create import *
from csalt.simulate import *
from csalt.infer import *
from csalt.utils2 import *
from csalt.data2 import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'

# controls
do_sim = False
do_inf = True

data_name = 'vet_approxs.ms'

if do_sim:
<<<<<<< HEAD
#    csim = create()
#    cfg_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'
#    cfg = [cfg_dir+'alma.cycle8.4.cfg', cfg_dir+'alma.cycle8.7.cfg']
#    _ = csim.template_MS(cfg, ['10min', '40min'], 'simtests/multi.ms',
#                         date=['2023/03/20', '2023/06/20'])
=======
    csim = create()
    cfg_dir = ''
    cfg = [cfg_dir+'alma.cycle8.4.cfg']#, cfg_dir+'alma.cycle8.7.cfg']
    _ = csim.template_MS(cfg, ['10min'], 'simtests/single.ms',
                         date=['2023/03/20'])#, '2023/06/20'])
>>>>>>> 0cd3ae3fab9e409cb1fd97787b0c7c55526538fb

    sim = simulate('CSALT')
    data_dict = read_MS('simtests/single.ms')
    inp = importlib.import_module('gen_fiducial_std')
    fixed_kw = {'FOV': inp.FOV[0], 'Npix': inp.Npix[0], 'dist': inp.dist,
                'Nup': 1, 'doppcorr': 'approx'}
    mdl_dict = sim.model(data_dict, inp.pars, kwargs=fixed_kw)
    write_MS(mdl_dict, outfile=data_name)

if do_inf:
    cinf = infer('CSALT')
    data_ = cinf.fitdata(data_name, vra=[-2000, 2200], chbin=1)
    inp = importlib.import_module('gen_fiducial_std')
    fixed_kw = {'FOV': inp.FOV[0], 'Npix': inp.Npix[0], 'dist': inp.dist,
                'Nup': 1, 'doppcorr': 'approx'}
    chi2 = -2 * cinf.log_likelihood(inp.pars, fdata=data_, kwargs=fixed_kw)
    print(chi2)


    lnprob, lnpri = cinf.log_posterior(inp.pars, fdata=data_, kwargs=fixed_kw)
    print(lnprob, lnpri)



sys.exit()


# create a measurement set from scratch
csim = create()

cfg_dir = '/pool/asha0/casa-release-5.7.2-4.el7/data/alma/simmos/'
cfg = [cfg_dir+'alma.cycle8.4.cfg', cfg_dir+'alma.cycle8.7.cfg']

_ = csim.template_MS(cfg, ['10min', '40min'], 'simtests/multi.ms',
                     date=['2023/03/20', '2023/06/20'])



# simulate a disk model on that MS
sim = simulate('CSALT')

msfile = 'simtests/multi.ms'
data_dict = read_MS(msfile)

inp = importlib.import_module('gen_fiducial_std')

mdl_dict = sim.model(data_dict, inp.pars, FOV=inp.FOV[0], Npix=inp.Npix[0],
                     dist=inp.dist, Nup=inp.nover)

write_MS(mdl_dict, outfile='test_multi.ms')




# CLEAN the cube
clean_kw = {'start': '-5.00km/s', 'width': '0.16km/s', 'nchan': 125,
            'imsize': 512, 'cell': '0.01arcsec'}
mask_kw = {'inc': 40, 'PA': 130, 'mstar': 1.0, 'zr': 0.3, 'r_max': 2.0, 
           'nbeams': 1.5, 'vlsr': 5.0e3}
imagecube('test_multi.ms', 'test_multi', mk_kepmask=True, 
          kepmask_kwargs=mask_kw, tclean_kwargs=clean_kw)
