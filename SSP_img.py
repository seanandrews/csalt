import os, sys, importlib
from csalt.utils import *
sys.path.append('configs/')

cfg = ['sg_taper2hi']

do_raw = True
raw_tau_only = False
do_sim = True
templ = False


# Loop through and create data, image the pure and noisy cubes
for i in range(len(cfg)):

    # load config file
    inp = importlib.import_module('gen_'+cfg[i])

    msname = inp.reduced_dir+inp.basename+'/'+inp.basename+'_pure.DATA'
    
    img_cube(msname, 
             inp.reduced_dir+inp.basename+'/images/'+ \
             inp.basename+'_pure.DATA', 'gen_'+cfg[i], masktype='kep')


