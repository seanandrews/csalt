import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *

# Read in the data MS
sdir = 'storage/data/DECO_tests/'
#ddict = read_MS('Sz129_200ch.ms') 

# Instantiate a csalt model
cm = model('CSALT_DECO_taper_1_c')

# Define some fixed attributes for the modeling
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 160, 'online_avg': 2} 

# Sample the posteriors!
_ = cm.sample_posteriors(sdir+'Sz129_200ch.ms', kwargs=fixed_kw,
                         vra=[1300, 6800], restfreq=230.538e9, 
                         Nwalk=75, Nthreads=8, Ninits=10, Nsteps=50,
                         outpost=sdir+'Sz129_posteriors.h5')
