import os
import sys
import numpy as np
from csalt.model import *
from csalt.helpers import *

# Read in the data MS
sdir = 'storage/data/DECO_tests/'

# Instantiate a csalt model
cm = model('CSALT_DECO_taper_1_c')

# Define some fixed attributes for the modeling
fixed_kw = {'FOV': 5.11, 'Npix': 512, 'dist': 160, 'online_avg': 2} 

# Sample the posteriors!
_ = cm.sample_posteriors(sdir+'Sz129_200ch.ms', kwargs=fixed_kw,
                         vra=[1300, 6800], restfreq=230.538e9, 
                         Nwalk=75, Nthreads=8, Ninits=10, Nsteps=50,
                         outpost=sdir+'Sz129_posteriors.h5')


# Analyze the posteriors

# look at the autocorrelation time evolution
_ = autocorr_evol_plot(sdir+'Sz129_posteriors.h5')


# load the burned-in, thinned, flattened posteriors
chain, logpost, logpri = load_posteriors(sdir+'Sz129_posteriors.h5',
                                         maxtau=10, burnfact=1, thinfact=0.5)


# make a trace plot
lbls = ['$i$', 'PA', '$M_{\\ast}$', '$R_{\\ell}^{\\sf f}$', 
        '$R_{\\ell}^{\\sf b}$', '$z_0^{\\sf f}$', '$\\psi^{\\sf f}$', 
        '$z_0^{\\sf b}$', '$\\psi^{\\sf b}$', '$T_0^{\\sf f}$', '$q^{\\sf f}$',
        '$T_0^{\\sf b}$', '$q^{\\sf b}$', '$\\phi^{\\sf f}$', 
        '$\\gamma^{\\sf f}$', '$\\phi^{\\sf b}$', '$\\gamma^{\\sf b}$', 
        '$\log_{10}{\\tau}$', '$p$', '$v_{\\rm sys}$', '$\\Delta x$', 
        '$\\Delta y$'] 
_ = trace_plot(sdir+'Sz129_posteriors.h5', labels=lbls)

