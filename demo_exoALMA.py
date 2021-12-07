import os, sys, importlib
from csalt.synthesize import make_template, make_data
from csalt.utils import cubestats, img_cube
sys.path.append('configs/')


cfg = 'FITStest'


# only need to do this **one time**.
make_template(cfg)

# impose a model onto the synthetic tracks
make_data(cfg, mtype='FITS', new_template=False)

# image the cube
inp = importlib.import_module('gen_'+cfg)
img_cube(inp.reduced_dir+inp.basename+'/'+inp.basename+'.noisy',
         inp.reduced_dir+inp.basename+'/images/'+inp.basename+'.noisy',
         'gen_'+cfg, masktype='kep')
