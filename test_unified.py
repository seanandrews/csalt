import os, sys, importlib, time
sys.path.append('configs/')
import numpy as np
from csalt.model import *
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['backend'] = 'TkAgg'


# Instantiate a csalt object
cm = model('CSALT')

# Create a fake dataset from scratch
#cfg_dir = ''
#cfg = [cfg_dir+'alma.cycle8.7.cfg']
#cm.template_MS(cfg, ['15min'], 'simtests/uber.ms', date=['2023/04/20'])

# Get the data dictionary from a MS
data_dict = cm.read_MS('simtests/uber.ms')

# Define the model parameters and keywords
inp = importlib.import_module('gen_fiducial_std')
fixed_kw = {'FOV': inp.FOV[0], 'Npix': inp.Npix[0], 'dist': inp.dist,
            'Nup': 1, 'doppcorr': 'approx'}

# Calculate a model dictionary
mdl_dict = cm.modeldict(data_dict, inp.pars, kwargs=fixed_kw)

# Write the model to a MS
cm.write_MS(mdl_dict, outfile='vet_unified.ms')

# Image a cube from that MS file
mask_kw = {'inc': inp.incl, 'PA': inp.PA, 'mstar': inp.mstar,
           'dist': inp.dist, 'vlsr': inp.Vsys}
clean_kw = {'start': '2km/s', 'width': '0.16km/s', 'nchan': 50}
#_ = imagecube('vet_unified.ms', 'vet_unified', 
#              tclean_kwargs=clean_kw, mk_kepmask=True, kepmask_kwargs=mask_kw)


# dummy cube

# Calculate a single model likelihood (or chi2 value)
fdata = cm.fitdata('vet_unified.ms', vra=[3000, 7000])
t0 = time.time()
chi2 = -2. * cm.log_likelihood(inp.pars, fdata=fdata, kwargs=fixed_kw)
print(time.time()-t0)
print(chi2)

print(inp.pars)


# sample the posteriors!
_ = cm.sample_posteriors('vet_unified.ms', vra=[3000, 7000], kwargs=fixed_kw,
                         Nthreads=6, Ninits=3, Nsteps=3)

# Sample the posteriors!

