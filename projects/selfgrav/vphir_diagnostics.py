import os, sys, importlib, time
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from eddy import linecube
import matplotlib.pyplot as plt

from matplotlib import cm, font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


# user-specified model
mdl = 'taper2hi'
dtyp = 'raw'

# additional controls
#

# load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+mdl)


# true surface
psurf = np.load('data/'+mdl+'.zCO_true.npz')['psurf']


# extract the velocity profile from the cube!
if dtyp == 'raw':
    cube = linecube(inp.radmcname+'raw_cube.fits', FOV=5.0)
    cube.data += np.random.normal(0, 1e-10, np.shape(cube.data))
else:
    cube = linecube(inp.reduced_dir+inp.basename+'/images/'+\
                    inp.basename+'_'+dtyp+'.DATA.image.fits', FOV=5.0)


ann1 = cube.get_annulus(r_min=0.775, r_max=0.825, inc=inp.incl, PA=inp.PA,
                        z0=psurf[0], psi=psurf[1], r_taper=psurf[2], 
                        q_taper=psurf[3], r_cavity=0)

fig = ann1.plot_river(return_fig=True)

# save figure to output
fig.savefig('figs/'+mdl+'_'+dtyp+'_river.jpg', dpi=300)
fig.clf()

print(ann1.get_vlos_SHO())

ann1.get_vlos_SHO()
fig = ann1.plot_river(vrot=ann1.get_vlos_SHO()[0][0], return_fig=True)

fig.savefig('figs/'+mdl+'_'+dtyp+'_straightriver.jpg', dpi=300)
fig.clf()
