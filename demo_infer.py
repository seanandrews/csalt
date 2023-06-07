import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *

# Read in the data MS
ddict = read_MS('testdata/Sz129_CO.ms') 

# Instantiate a csalt model
cm = model('CSALT')

# Define some fixed attributes for the modeling
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 161} 

# Sample the posteriors!
_ = cm.sample_posteriors('testdata/Sz129_CO.ms', kwargs=fixed_kw,
                         vra=[2300, 4500], restfreq=230.538e9, 
                         Nwalkers=75, Nthreads=6, Ninits=10, Nsteps=50,
                         outpost='testdata/Sz129_posteriors.h5')

