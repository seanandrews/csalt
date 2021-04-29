"""
decription tbd

"""

import os, sys, time
import numpy as np
import const as c_
import mconfig as in_
from cube_parser import cube_parser


# Generate a synthetic observation template
gen_template = False
if in_.gen_uv:
    # First, check if the template exists; decide whether to use it
    if np.logical_and(os.path.exists('obs_templates/'+in_.template+'.uvfits'),
                      os.path.exists('obs_templates/'+in_.template+'.params')):
       io = np.loadtxt('obs_templates/'+in_.template+'.params', dtype=str)
       io_in = [in_.RA, in_.DEC, in_.date, in_.HA, in_.config, in_.ttotal,
                in_.integ]
       if io == io_in:
           print('This template exists.  Would you like to use the one on ' + 
                 'file already? (y | n)')
           use_existing = input()
           if np.logical_or((input() == 'y'), (input() == 'Y')):
               gen_template = False
           else:
               os.system('rm -rf obs_templates/'+in_.template+'*')
               os.system('rm -rf obs_templates/sims/'+in_.template+'*')
               gen_template = True
    else:
        gen_template = True
    
if gen_template: 
    # Set the channels in frequency and LSRK velocity domains
    nu_span = in_.restfreq * (in_.vsys - in_.vspan) / c_.c
    nu_sys  = in_.restfreq * (1 - in_.vsys / c_.c)
    nch = np.int(2 * np.abs(nu_span) / (in_.dfreq0 / in_.sosampl) + 1)
    freq = nu_sys - nu_span - (in_.dfreq0 / in_.sosampl) * np.arange(nch)
    vel = c_.c * (1. - freq / in_.restfreq)

    # Parse the target coordinates into degrees
    RA_pieces = [np.float(in_.RA.split(':')[i]) for i in np.arange(3)]
    RAdeg = 15 * np.sum(np.array(RA_pieces) / [1., 60., 3600.])
    DEC_pieces = [np.float(in_.DEC.split(':')[i]) for i in np.arange(3)]
    DECdeg = np.sum(np.array(DEC_pieces) / [1., 60., 3600.])

    # Generate a model cube; save it as a FITS file
    cube_parser(in_.pars, FOV=8, Npix=256, dist=150, r_max=300, 
                restfreq=in_.restfreq, RA=RAdeg, DEC=DECdeg, Vsys=in_.vsys,
                vel=vel, outfile='obs_templates/'+in_.template+'.fits')

    # Generate the (u,v) tracks
    os.system('casa --nologger --nologfile -c CASA_scripts/mock_obs.py')
