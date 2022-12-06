import os, sys, importlib
from csalt.synthesize import make_template, make_data
from csalt.utils import img_cube, cubestats
sys.path.append('configs/')


cfg = 'csalt2_exoalma'

# only need to do this **one time**.
#make_template(cfg)

# impose a model onto the synthetic tracks
make_data(cfg, mtype='CSALT', new_template=False)

# image the cubes
inp = importlib.import_module('gen_'+cfg)
img_cube(inp.reduced_dir+inp.basename+'/'+inp.basename+'_noisy.DATA',
         inp.reduced_dir+inp.basename+'/images/'+inp.basename+'_noisy.DATA',
         'gen_'+cfg, masktype='kep')

img_cube(inp.reduced_dir+inp.basename+'/'+inp.basename+'_pure.DATA',
         inp.reduced_dir+inp.basename+'/images/'+inp.basename+'_pure.DATA',
         'gen_'+cfg, masktype='kep')


cubestats(inp.reduced_dir+inp.basename+'/images/'+inp.basename+'_noisy.DATA')
